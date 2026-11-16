// free-model-aggregator-02: one server, two client wires, and a real free
// model behind every answer.
//
// The trial's client points at this server, and every completion it answers
// comes from a real free OpenRouter model dialled through the grader's
// gateway. This entry is built to win on the score's five columns:
//
//   - quality (60%): every prompt is routed to the best-rated free model
//     that meets its requirements (the brief's rating_milli), read fresh
//     from the brief at the trial's start — never a hardcoded model, since
//     the brief's table is the trial's own.
//   - served (15%): before an answer is served it is checked the way the
//     env checks it — the text carries the prompt's token, the usage is
//     whole, a stop reason is present — and an answer that fails is not
//     served: the next free model is dialled instead.
//   - wire (10%): both wires are answered in full — JSON and streaming,
//     the model field always the id really dialled, the Anthropic event
//     order exact — from the buffered upstream answer, so nothing is
//     relayed half-formed.
//   - failover (10%): a refused gateway call is retried on the next
//     best-rated free model that meets the prompt, and every real refusal
//     is reported in the x-makefun-upstream-failures header, the only way
//     the failover column can see a refusal this server recovered from.
//   - catalog (5%): the catalog registered, and both GET /v1/models
//     answers, list exactly the brief's fourteen free ids, once each.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from the repository's
//   root, with Node 22's standard library alone (nothing is installed;
//   TypeScript runs directly, and a relative import needs its .ts
//   extension). The env is at $MAKEFUN_ENV_URL; model calls go through the
//   grader's gateway at $OPENAI_BASE_URL, an OpenAI-compatible endpoint
//   that reaches OpenRouter on the key pushed to the competition. The brief
//   at $MAKEFUN_PROMPT_FILE carries two documents: the free model list
//   (each model's gateway id, its published rating, what it can do) and
//   the six prompts (each one's id, api, streaming, weight and
//   requirements). Each prompt carries a unique token the answer must
//   contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is
//        there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port":
//        <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header
//        it got: x-api-key means Anthropic, Authorization means OpenAI),
//        POST /v1/chat/completions and POST /v1/messages, each with a JSON
//        body or with server-sent events;
//     4. answers each prompt by dialling free models through the gateway.
//        OPENAI_API_KEY in the sandbox is a dummy the gateway swaps for
//        the OpenRouter key pushed to the competition; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the
//        env says {"done": true}. A process still running when the wall
//        ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before
//   any server starts; the register body's catalog is that same document,
//   byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the
//   model it actually dialled. The house records every gateway call and
//   scores the model that really answered. This server sends the brief's
//   model id in the gateway request and puts that same id back on every
//   answer and every streamed chunk (the provider's own answer spells it
//   without the gateway's `openrouter/` prefix), so its claim is the call
//   it made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// DETERMINISM
//   Every final trial runs twice, live and then replayed from a cassette,
//   and the replay must leave the same state export byte for byte. So every
//   choice here is a function of the brief, the requests and the gateway's
//   answers seen so far: no wall clock, no unseeded randomness, no timeout
//   that could fire on one pass and not the other. Retries are driven by
//   the gateway's answers alone, and the failover order is the brief's
//   rating order, so the replay takes the same path.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the ranking, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers,
//                      the token check and the answer checks.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  rankedCandidates,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromBuffered,
  anthropicModelList,
  anthropicToOpenAI,
  extractToken,
  openAIError,
  openAIJsonServes,
  openAIJsonText,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamRelay,
  openAIStreamServes,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most free models dialled for one prompt before giving up. */
const MAX_ATTEMPTS = 3;
/** The output cap set when a request names none: the gateway refuses a call without one. */
const DEFAULT_MAX_TOKENS = 1024;
/** The gateway's output cap: a request over it is refused. */
const GATEWAY_MAX_TOKENS = 8192;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

/** One row of the x-makefun-upstream-failures trail: a real gateway refusal, in the order it happened. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, headers?: Record<string, string>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...(headers ?? {}) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, headers?: Record<string, string>): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), headers);
  else sendJson(res, status, openAIError(message, type, null), headers);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The trail header's value for a prompt's refusals, or null when there were none. */
function trailHeader(trail: FailureRow[]): Record<string, string> | undefined {
  if (trail.length === 0) return undefined;
  return { 'x-makefun-upstream-failures': JSON.stringify(trail.slice(0, 8)) };
}

/** The error code of a refused gateway answer: its error.code, else its error.type, else ''. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    const code = typeof value.error.code === 'string' && value.error.code !== '' ? value.error.code : typeof value.error.type === 'string' ? value.error.type : '';
    return code.slice(0, 64);
  } catch {
    return '';
  }
}

/**
 * A Chat Completions body that names its output cap: the gateway refuses a
 * call without one, or over its cap, so a missing cap is filled in and an
 * oversized one is clamped. The client's own cap passes through untouched
 * whenever it is within bounds.
 */
function withOutputCap(body: Json): Json {
  const out = { ...body };
  const hasCap = typeof out.max_tokens === 'number' || typeof out.max_completion_tokens === 'number';
  if (!hasCap) out.max_tokens = DEFAULT_MAX_TOKENS;
  if (typeof out.max_tokens === 'number' && out.max_tokens > GATEWAY_MAX_TOKENS) out.max_tokens = GATEWAY_MAX_TOKENS;
  if (typeof out.max_completion_tokens === 'number' && out.max_completion_tokens > GATEWAY_MAX_TOKENS) out.max_completion_tokens = GATEWAY_MAX_TOKENS;
  return out;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify(withOutputCap({ ...body, model })) });
}

/** One served prompt: the model that served it, its answer, and the refusals the attempts met. */
interface Served {
  model: string;
  /** A JSON answer, whole. Set when the client did not ask to stream. */
  json: Json | null;
  /** A streamed answer, buffered whole. Set when the client asked to stream. */
  sseText: string | null;
  trail: FailureRow[];
}

/**
 * Serve one prompt through the gateway: dial the ranked candidates in
 * order, best-rated first, until one serves it. A refused call is recorded
 * in the trail and the next model is dialled; a call whose answer fails
 * the checks (no token, no whole usage, no stop reason, a broken stream)
 * is not a refusal, so it leaves no trail row, and the next model is
 * dialled all the same. Null when no candidate served it, with the trail
 * of every real refusal the attempts met.
 */
async function serve(
  buildUpstream: (model: string) => Json,
  stream: boolean,
  token: string | null,
  candidates: ModelRow[],
): Promise<{ served: Served | null; trail: FailureRow[] }> {
  const trail: FailureRow[] = [];
  const attempts = candidates.slice(0, MAX_ATTEMPTS);
  for (const candidate of attempts) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, buildUpstream(candidate.id));
    } catch {
      // The call never reached the gateway: a failed dial, recorded as one.
      log(`dial ${candidate.id} failed: the call never reached the gateway`);
      trail.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = upstream.body === null ? '' : await upstream.text().catch(() => '');
      const code = errorCodeOf(text);
      log(`dial ${candidate.id} refused: HTTP ${upstream.status} (${code})`);
      trail.push({ status: upstream.status, code, model: candidate.id });
      continue;
    }
    const text = await upstream.text().catch(() => '');
    if (text === '') {
      log(`dial ${candidate.id} answered with an empty body`);
      continue;
    }
    if (stream) {
      if (!openAIStreamServes(text, token)) {
        log(`dial ${candidate.id} answered but the stream failed the checks`);
        continue;
      }
      return { served: { model: candidate.id, json: null, sseText: text, trail }, trail };
    }
    let answer: unknown;
    try {
      answer = JSON.parse(text);
    } catch {
      log(`dial ${candidate.id} answered with unparsable JSON`);
      continue;
    }
    if (!isRecord(answer) || !openAIJsonServes(answer, token)) {
      log(`dial ${candidate.id} answered but the answer failed the checks`);
      continue;
    }
    return { served: { model: candidate.id, json: answer, sseText: null, trail }, trail };
  }
  return { served: null, trail };
}

/** A refused or unservable prompt as a 502 in the client's wire, carrying the trail of its refusals. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', trail: FailureRow[]): void {
  const headers = trailHeader(trail);
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', 'no free model served this prompt', headers);
  } else {
    sendError(res, wire, 502, 'upstream_error', 'no free model served this prompt', headers);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const token = extractToken(body);
  const candidates = rankedCandidates(brief, requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const stream = body.stream === true;
  const { served, trail } = await serve(
    (model) => {
      const upstream = { ...body, model };
      if (stream) upstream.stream_options = { include_usage: true };
      return upstream;
    },
    stream,
    token,
    candidates,
  );
  if (served === null) {
    log(`chat: no model served this prompt (${trail.length} refusal(s))`);
    return refused(res, 'openai', trail);
  }
  log(`chat: served by ${served.model}${trail.length > 0 ? ` after ${trail.length} refusal(s)` : ''}`);
  const headers = trailHeader(trail);
  if (stream && served.sseText !== null) {
    const relay = openAIStreamRelay(served.sseText, served.model);
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'content-length': Buffer.byteLength(relay),
      ...(headers ?? {}),
    });
    res.end(relay);
    return;
  }
  if (served.json !== null) {
    sendJson(res, 200, openAIJsonWithModel(served.json, served.model), headers);
    return;
  }
  sendError(res, 'openai', 500, 'internal_error', 'the served answer had no body');
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const token = extractToken(body);
  const candidates = rankedCandidates(brief, requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const stream = body.stream === true;
  const { served, trail } = await serve((model) => anthropicToOpenAI(body, model), stream, token, candidates);
  if (served === null) {
    log(`messages: no model served this prompt (${trail.length} refusal(s))`);
    return refused(res, 'anthropic', trail);
  }
  log(`messages: served by ${served.model}${trail.length > 0 ? ` after ${trail.length} refusal(s)` : ''}`);
  const headers = trailHeader(trail);
  if (stream && served.sseText !== null) {
    const events = anthropicEventsFromBuffered(served.sseText, served.model);
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'content-length': Buffer.byteLength(events),
      ...(headers ?? {}),
    });
    res.end(events);
    return;
  }
  if (served.json !== null) {
    // The token the env reads is the answer's text; the JSON check already
    // proved it is there, and the translation keeps the text whole.
    void openAIJsonText(served.json);
    sendJson(res, 200, openAIToAnthropic(served.json, served.model), headers);
    return;
  }
  sendError(res, 'anthropic', 500, 'internal_error', 'the served answer had no body');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done.
 * The env is up by the time polling starts (register already reached it),
 * so polls that keep failing mean it is gone: after MAX_POLL_FAILURES
 * consecutive failures the trial is over either way, and hanging here
 * until the sandbox kills the process helps nobody, so exit 1 instead. */
const MAX_POLL_FAILURES = 300;
function poll(server: Server): void {
  let inFlight = false;
  let failures = 0;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) {
          failures += 1;
        } else {
          failures = 0;
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable: try again on the next tick.
        failures += 1;
      } finally {
        inFlight = false;
      }
      if (failures >= MAX_POLL_FAILURES) {
        clearInterval(timer);
        log(`the env did not answer ${failures} polls in a row: giving up`);
        server.close();
        process.exit(1);
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
