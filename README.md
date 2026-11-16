# makefun-free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/ssj6go/makefun-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/ssj6go/makefun-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator: one server answers both the OpenAI and the Anthropic completion wires, and every answer comes from a real free OpenRouter model dialled through the grader's gateway. Each prompt is routed to the best-rated free model that meets its requirements, retried on the next-best when a call is refused or its answer fails the checks, and every real refusal is reported in the `x-makefun-upstream-failures` header.

## Run it

```sh
node main.ts --list
```

```sh
MAKEFUN_ENV_URL=http://localhost:8080 MAKEFUN_PROMPT_FILE=prompt.txt OPENAI_BASE_URL=http://localhost:8080/v1 OPENAI_API_KEY=<trial key> node main.ts
```

## How it works

- `main.ts` — the server: the brief, the catalog, the routes, the register, the work poll, and the routing with failover.
- `lib/providers.ts` — the brief's parser, the free model list, the rating-ranked candidate order, and the catalog document.
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation, the model field, the SSE writers, the prompt-token check, and the answer checks.

Routing is rating-first: the brief's `rating_milli` decides which free model serves each prompt, and the failover order is the same ranking. Before an answer is served it is checked the way the trial's env checks it — the text carries the prompt's token, the usage is whole, a stop reason is present, and a stream ends with a finish chunk and `[DONE]` — so a model that answers without the token costs one retry instead of the prompt. Streams are buffered whole before they are checked and relayed. Every choice is a function of the brief, the requests and the gateway's answers: no wall clock and no unseeded randomness, so a replayed trial takes the same path.

---

<sub>Entered on [make.fun](https://make.fun/?ref=yk2cnwj9&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
