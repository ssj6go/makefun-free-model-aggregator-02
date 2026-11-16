// The client wires, the one upstream wire, the model field, and the answer
// checks.
//
// The trial's client calls the entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. The entry dials the grader's gateway, an
// OpenAI-compatible endpoint, so every prompt goes upstream as a Chat
// Completions request and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id)
//     to the gateway and relays the gateway's JSON or SSE answer, with every
//     `model` field set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system
//     prompt, image blocks, tool definitions, tool results and tool choice
//     included) and rewrites the gateway's answer into a Messages object, or
//     into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the
//     stop reason and the usage translated. Tool calls become tool_use blocks
//     and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really
//   answered, so a response's `model` must be the model this entry dialled.
//   OpenRouter's own answer names a free model without the gateway's
//   `openrouter/` prefix; these functions put the dialled id back on every
//   answer and every streamed chunk. They never make a model up: the id came
//   from the brief and went to the gateway in the request's `model`.
//
// THE ANSWER CHECKS
//   A prompt earns nothing unless the answer the client sees is valid and
//   the provider's own body carries the prompt's token, so before an answer
//   is served this file checks it the way the env does: the text carries the
//   token, the usage is whole (an input count of at least 0 and an output
//   count of at least 1), a stop reason the API knows is present, and, for a
//   stream, a finish chunk and a [DONE] terminator. An answer that fails any
//   of these is not served: the caller dials the next free model instead.
//   Streams are buffered whole before they are checked and relayed, so the
//   check sees the full answer; the client waits up to 30 seconds per
//   answer, and these short completions stream in seconds.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same gateway answer give the same
// bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** A prompt's token, as the env spells it: WORD-1234, the word 3 to 12 capitals. */
const TOKEN_RE = /\b([A-Z]{3,12}-[0-9]{4})\b/;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** Every text string a request body carries: the system prompt and each message's text. */
function requestTexts(body: Json): string[] {
  const out: string[] = [];
  const system = textOf(body.system);
  if (system !== '') out.push(system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const text = textOf(message.content);
    if (text !== '') out.push(text);
  }
  return out;
}

/**
 * The prompt's token, read off the request body: the instruction carries
 * exactly one token (WORD-1234), and the answer must carry it back. Null
 * when the body names none, in which case the answer is checked on its
 * usage and stop reason alone.
 */
export function extractToken(body: Json): string | null {
  for (const text of requestTexts(body)) {
    const found = TOKEN_RE.exec(text);
    if (found !== null) return found[1];
  }
  return null;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
export function openAIJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

/** The text the env reads from a Chat Completions JSON answer: the first choice's message content. */
export function openAIJsonText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  return typeof message.content === 'string' ? message.content : textOf(message.content);
}

/**
 * Whether a Chat Completions JSON answer is worth serving: the first choice
 * is the assistant's with a finish reason, the usage is whole (an input
 * count of at least 0 and an output count of at least 1), and, when the
 * prompt's token is known, the answer's text carries it. The checks mirror
 * the env's own, so an answer that passes here passes there.
 */
export function openAIJsonServes(body: Json, token: string | null): boolean {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  if (choices.length === 0) return false;
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return false;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null || message.role !== 'assistant') return false;
  if (typeof choice.finish_reason !== 'string' || choice.finish_reason === '') return false;
  const usage = isRecord(body.usage) ? body.usage : null;
  if (usage === null) return false;
  const input = numberOrNull(usage.prompt_tokens);
  const output = numberOrNull(usage.completion_tokens);
  if (input === null || output === null || input < 0 || output < 1) return false;
  if (token !== null && !openAIJsonText(body).includes(token)) return false;
  return true;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * One SSE frame with every data payload's `model` set to `model`, and its
 * blank line kept: comments, `event:` lines and payloads that are not JSON
 * objects pass through untouched. Chat Completions names the model on its
 * chunks; the client sees the id the entry dialled, never the provider's
 * spelling.
 */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/** Split buffered SSE text into frames, as the SSE grammar does (blank line ends a frame). */
export function sseFrames(text: string): string[] {
  const normalized = text.replace(/\r\n/g, '\n');
  const frames: string[] = [];
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() !== '') frames.push(frame);
  }
  return frames;
}

/** A buffered Chat Completions SSE stream, read the way the env reads it. */
export interface StreamReading {
  /** The concatenated text of every content delta. */
  text: string;
  /** The last model id any chunk named. */
  model: string | null;
  /** The usage the final chunk carried, or null when no chunk carried one. */
  usage: { input: number | null; output: number | null } | null;
  /** The finish reason the finish chunk named, or null when there was none. */
  finishReason: string | null;
  /** Whether a [DONE] frame closed the stream, as the last frame. */
  sawDone: boolean;
  /** Whether the stream is well-formed: every data payload a JSON object with choices. */
  shapeOk: boolean;
}

/** Read a buffered Chat Completions SSE stream: its text, usage, finish reason and terminator. */
export function readOpenAIStream(text: string): StreamReading {
  const frames = sseFrames(text);
  let joined = '';
  let model: string | null = null;
  let usage: StreamReading['usage'] = null;
  let finishReason: string | null = null;
  let sawDone = false;
  let shapeOk = frames.length > 0;
  let sawFinish = false;
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        if (i !== frames.length - 1) shapeOk = false;
        sawDone = true;
        continue;
      }
      if (sawDone) {
        shapeOk = false;
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        shapeOk = false;
        continue;
      }
      if (!isRecord(chunk) || !Array.isArray(chunk.choices)) {
        shapeOk = false;
        continue;
      }
      if (typeof chunk.model === 'string' && chunk.model !== '') model = chunk.model;
      if (isRecord(chunk.usage)) {
        usage = { input: numberOrNull(chunk.usage.prompt_tokens), output: numberOrNull(chunk.usage.completion_tokens) };
      }
      if (chunk.choices.length === 0) continue;
      const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
      if (choice === null) {
        shapeOk = false;
        continue;
      }
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (sawFinish) shapeOk = false;
        joined += delta.content;
      }
      if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') {
        if (sawFinish) shapeOk = false;
        sawFinish = true;
        finishReason = choice.finish_reason;
      }
    }
  }
  if (!sawFinish) shapeOk = false;
  if (!sawDone) shapeOk = false;
  return { text: joined, model, usage, finishReason, sawDone, shapeOk };
}

/**
 * Whether a buffered Chat Completions SSE stream is worth serving: it is
 * well-formed, it ends with [DONE] after a finish chunk, its usage is
 * whole, and, when the prompt's token is known, its text carries it.
 */
export function openAIStreamServes(text: string, token: string | null): boolean {
  const reading = readOpenAIStream(text);
  if (!reading.shapeOk) return false;
  if (reading.finishReason === null) return false;
  if (reading.usage === null) return false;
  const { input, output } = reading.usage;
  if (input === null || output === null || input < 0 || output < 1) return false;
  if (token !== null && !reading.text.includes(token)) return false;
  return true;
}

/** A buffered Chat Completions SSE stream relayed whole, every chunk's `model` the id dialled. */
export function openAIStreamRelay(text: string, model: string): string {
  return sseFrames(text)
    .map((frame) => frameWithModel(frame, model))
    .join('');
}

/**
 * A buffered Chat Completions SSE stream as the Anthropic event sequence:
 * message_start on the first chunk, content_block_start/delta/stop per text
 * block or tool call, then message_delta with the stop reason and usage,
 * then message_stop. `model` is the id the entry dialled, named back to
 * the client. The input is the whole buffered stream, so the output is one
 * deterministic document for one upstream answer.
 */
export function anthropicEventsFromBuffered(text: string, model: string): string {
  const out: string[] = [];
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): void => {
    if (started) return;
    started = true;
    out.push(
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    );
  };
  const closeEvents = (): void => {
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
  };
  const finishEvents = (stopReason: string): void => {
    if (finished) return;
    finished = true;
    closeEvents();
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): void => {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        startEvents();
        finishEvents(pendingStop ?? 'end_turn');
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      startEvents();
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
  };

  for (const frame of sseFrames(text)) eventsOfFrame(frame);
  if (!finished) {
    startEvents();
    finishEvents(pendingStop ?? 'end_turn');
  }
  return out.join('');
}
