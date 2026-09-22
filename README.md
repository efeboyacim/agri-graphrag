# AgriGraphRAG

A small but complete **agentic Graph RAG** demo for the agriculture domain. It answers questions from tomato growers in İzmir and Manisa. Each question is enriched with:

- structured knowledge from a **Neo4j** graph,
- free-text agronomy notes from **LanceDB**,
- mock social media posts.

Four cooperating agents combine these into one answer with source references. Every LLM call goes through the **Portkey** gateway. Answer quality is measured offline with a structured **DeepEval** test suite and a prompt A/B experiment.

The dataset is deliberately tiny: 20 graph nodes and 8 document chunks. The point is to prove the architecture end to end, not to be a production system.

## Architecture

```
                         POST /query  (FastAPI, app/main.py)
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 1. Orchestrator Agent   (gpt-oss-20b)                                        │
 │    intent + entities (crop, region, symptoms, disease) → which agents to run │
 └──────────────────────────────────────────────────────────────────────────────┘
                 │                                          │
                 ▼                                          ▼
 ┌──────────────────────────────────────────┐   ┌──────────────────────────────────┐
 │ 2. Retrieval Agent (no LLM)              │   │ 3. Social Listening Agent        │
 │  Neo4j Cypher: crop+symptom → disease    │   │    (gpt-oss-20b)                 │
 │                → product (+ region)      │   │  SocialPost nodes filtered by    │
 │  LanceDB vector search (MiniLM, local),  │   │  region → summary / trend        │
 │  query expanded with graph diseases      │   │                                  │
 │  → merged, ordered context [G#][D#]      │   │  → context [S#]                  │
 └──────────────────────────────────────────┘   └──────────────────────────────────┘
                 │                                          │
                 └────────────────────┬─────────────────────┘
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 4. Synthesis Agent   (gpt-oss-120b)                                          │
 │    grounded answer citing [G#] graph facts, [D#] notes, [S#] social posts    │
 └──────────────────────────────────────────────────────────────────────────────┘

 Every LLM call ──► app/llm_client.py ──► Portkey gateway (trace_id, span=step, metadata)
                                        ──► Groq
 Offline only:  eval/test_suite.py, eval/prompt_experiment.py ──► DeepEval (judge also via Portkey)
```

| Agent | File | Model | What it does |
|---|---|---|---|
| Orchestrator | `app/agents/orchestrator.py` | `SMALL_MODEL` | Returns a JSON `QueryPlan` with the intent, crop, region, symptoms, keywords and a mentioned disease, and decides which sub-agents run. Deterministic guardrails then map the region to a graph `Region` node. This handles Turkish dotted İ: `"İzmir".lower()` ≠ `"izmir"`. |
| Retrieval | `app/agents/retrieval.py` | none | **Hybrid retrieval.** A Cypher query scores diseases by symptom-keyword overlap and returns disease, symptoms, product and region. Trend questions fall back to a Region ← SocialPost → Disease traversal. LanceDB returns the top 3 notes by cosine distance, using the query expanded with the graph's disease names. Output is merged into an ordered context: graph facts first, then notes. |
| Social Listening | `app/agents/social_listening.py` | `SMALL_MODEL` | Reads `SocialPost` nodes for the region and summarises the trend and concern level. |
| Synthesis | `app/agents/synthesis.py` | `LARGE_MODEL` | Writes the final answer, citing context ids. Has two prompt versions, `v1` (naive) and `v2` (grounded, production). |

`app/pipeline.py` is the one flow function used by `/query`, `test_e2e.py` and both eval scripts, so all of them run exactly the same code.

### Models: why not Llama?
The original design used `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`. Groq **retired both for free and developer accounts on 2026-08-16**. The demo therefore defaults to Groq's recommended replacements:

- `openai/gpt-oss-20b` for the small tasks
- `openai/gpt-oss-120b` for synthesis and the DeepEval judge

All three are configurable in `.env` (`SMALL_MODEL`, `LARGE_MODEL`, `JUDGE_MODEL`). gpt-oss models reason before answering, so the client sends `reasoning_effort="low"` and a generous token cap.

## Why Portkey is mandatory
LLM agents are hard to debug without per-call visibility. Every LLM call goes through `app/llm_client.py`, which sends it to Groq through the Portkey gateway. That covers the orchestrator, social listening, synthesis and the DeepEval judge, with no exceptions. Each call carries:

- `trace_id`: one per user query, so all agent calls of a query form **one trace** in Portkey.
- `span_id` and `span_name`: the agent step (`orchestrator`, `social_listening`, `synthesis`, `judge.faithfulness`, …).
- `metadata`: `app`, `step`, `agent`, `model`, and extras such as `prompt_version` or `region`.

**Portkey records the model, latency, tokens and cost** of every request it proxies. The same numbers are mirrored locally (`llm_calls` in the `/query` response and the `test_e2e.py` trace table), because the free tier has no API to read cost back.

The gateway also echoes the trace id in a response header. `test_e2e.py` prints it as proof that the call really went through Portkey.

**How Portkey reaches Groq.** There are two modes, set in `.env`:
- **Saved integration (recommended).** Add Groq once in Portkey's *Model Catalog*; your Groq key is stored in Portkey. Then set `PORTKEY_PROVIDER=@<slug>`, e.g. `@groq`. This mode is required if the workspace enforces *block inline config*. In that case Portkey rejects both inline configs and inline provider names with `inline_config_blocked` or `inline_provider_blocked`.
- **Inline provider (default).** Leave `PORTKEY_PROVIDER` unset. Portkey then forwards `GROQ_API_KEY` from `.env`.

Gateway retries are optional through a saved config (`PORTKEY_CONFIG=pc-...`). The client always retries Groq 429s and 5xx errors itself, honouring `retry-after`.

**Fail fast:** at startup the app checks that `PORTKEY_API_KEY` (and `GROQ_API_KEY` in inline mode) are set and are not the placeholders. If either is missing, it raises and exits:

```
PortkeyNotConfiguredError: PORTKEY_API_KEY not set. AgriGraphRAG routes every LLM call through the Portkey gateway ... and will not start without it; there is no fallback to direct Groq calls or console-only logging.
```

## Setup

Prerequisites: Docker Desktop and Python 3.10+. You also need a Groq API key and a Portkey API key; both have free tiers.

```bash
cp .env.example .env              # PowerShell: Copy-Item .env.example .env
# edit .env: set PORTKEY_API_KEY (REQUIRED) and either PORTKEY_PROVIDER=@<your Groq integration slug>
# (saved integration in Portkey's Model Catalog) or GROQ_API_KEY (inline mode); see "How Portkey reaches Groq"
```

**1. Start the system.** This brings up Neo4j and the FastAPI app, each as a container:
```bash
docker compose up -d --build
curl http://localhost:8000/health
```
The first build downloads CPU-only torch and the MiniLM model, and bakes a LanceDB copy of the notes into the image.

> **Ports already in use?** For example, another Neo4j on 7474/7687 or another API on 8000. Pick different host ports in `.env`, and point the host scripts at the new bolt port:
> ```
> NEO4J_HTTP_PORT=17474
> NEO4J_BOLT_PORT=17687
> APP_PORT=8001
> NEO4J_URI=bolt://localhost:17687
> ```
> The app container always reaches Neo4j internally at `bolt://neo4j:7687`.

**2. Load the data.** Run this from the host after creating a virtualenv:
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python seed/seed_neo4j.py && python seed/seed_lancedb.py
```
Windows PowerShell 5.1 has no `&&`, so use `;` instead. `seed_neo4j.py` is idempotent and loads 20 nodes and 19 relationships. As a safety measure, it only wipes a database that is empty or that it seeded before; it leaves an `:AgriGraphRAGSeed` marker node. It refuses to touch any other non-empty Neo4j unless you pass `--force`. `seed_lancedb.py` embeds 8 notes into `./data/lancedb`, which is used by the host-side scripts.

> The app container keeps its own LanceDB copy inside the image, seeded at build time from the same script. That avoids a Windows bind mount, which is known to corrupt LanceDB datasets. After editing the notes, run `docker compose build app`.

**3. Ask a question:**
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"White spots showed up on my tomato leaves in İzmir\"}"
```
The response contains the `answer`, cited `sources`, `graph_nodes`, the merged `context`, the orchestrator `plan`, the `social` summary, and `llm_calls`. Each call lists its step, model, latency, tokens, cost, span id and gateway trace id. There are also `totals`. Interactive docs are at http://localhost:8000/docs.

**4. End-to-end demo:**
```bash
python test_e2e.py
```
This prints each agent's output step by step, then the Portkey trace table and PASS/FAIL checks. The answer must mention Powdery Mildew, FungiStop-X and the social trend, and Portkey must echo the trace id.

**Fail-fast check:**
```bash
docker compose run --rm -e PORTKEY_API_KEY= app     # exits with PortkeyNotConfiguredError
```

## Evaluation pipeline (DeepEval)

The evaluation pipeline is offline and runs at development time. It is **not** part of the live `/query` flow and never blocks a request. It has two parts.

### 1. Structured test suite: `eval/test_suite.py`
`eval/dataset.py` holds a fixed set of 6 representative questions. They span 4 tomato problems, both regions, a question with no region (social listening skipped) and a trend question. Each case has its **expected key facts** (disease and product) and a reference answer.

Every case runs through the full agent pipeline and is scored on four metrics:

| Metric | Type | What it measures |
|---|---|---|
| Faithfulness | LLM judge | Are the answer's claims supported by the retrieved context? Uses `penalize_ambiguous_claims=True`. By default DeepEval counts claims the context can't verify ("idk") as faithful, which would hide ungrounded general-knowledge claims. |
| Answer Relevancy | LLM judge | Does the answer address the question? |
| Contextual Precision | LLM judge | Is relevant context ranked above irrelevant context? This scores the retriever. |
| Key Facts | deterministic | The fraction of the expected disease and product names present in the answer. |

The LLM-judged metrics have a threshold of 0.7. The judge is `JUDGE_MODEL`, running through Portkey via `eval/metrics.py:PortkeyGroqJudge`.

```bash
python eval/test_suite.py                 # aggregate report: per-case table, mean + pass rate per metric, overall pass rate
python eval/test_suite.py --limit 2       # first N cases only
deepeval test run eval/test_suite.py      # same suite as a pytest test run (one test per case)
deepeval test run eval/test_suite.py -k pm_izmir_white_spots
```
Results are saved to `eval/results/test_suite_<timestamp>.json`.

### 2. Prompt experiment: `eval/prompt_experiment.py`
This compares two Synthesis Agent prompts on the same test set:

- **v1**, deliberately naive: "answer thoroughly from your agricultural knowledge". The context is loosely appended as "notes that may or may not be relevant", with no citation rules.
- **v2**, the production prompt: answer only from the context, cite `[G#]/[D#]/[S#]` for every claim, recommend only graph products, and say so when the context doesn't cover something.

For each case, the upstream agents run **once**, and both prompts synthesise from the identical context, so the prompt is the only variable.

Both versions are scored on faithfulness, answer relevancy and key facts. Contextual precision is left out because it depends only on retrieval, which is identical for both. Each version is a separate DeepEval run with `hyperparameters={"synthesis_prompt": v}`. If you set `CONFIDENT_API_KEY`, both runs also appear on Confident AI, where you can compare them side by side.

```bash
python eval/prompt_experiment.py
```
The script prints the side-by-side table (mean, pass rate, delta), per-case scores and a plain verdict, and saves the full results including both answers per case.

### Results (real runs, 2026-09-22; free-tier Groq via Portkey)

**Test suite, first full run** (v2 prompt as originally written): 4/6 cases passed all metrics.

| Metric | Mean | Pass rate |
|---|---|---|
| Faithfulness | 0.81 | 83% |
| Answer Relevancy | 0.91 | 83% |
| Contextual Precision | 0.94 | 100% |
| Key Facts | 1.00 | 100% |

The suite caught two real weaknesses, and v2 was refined once in response:

| Case | Problem found | Before → after |
|---|---|---|
| `early_blight_manisa` | The answer added uncited "common knowledge" advice (drip irrigation, crop rotation). With `penalize_ambiguous_claims` these claims count as unfaithful. v2 now says: if a sentence cannot be cited, leave it out. | faithfulness 0.38 → 1.00 |
| `manisa_trends` | A fixed diagnosis template forced treatment and prevention advice into a "what are growers reporting?" question. v2 now adapts its sections to the question type. | answer relevancy 0.43 → 1.00 |

The same run surfaced two formatting issues, both now normalised by the LLM client:
- gpt-oss wrote product names with non-breaking hyphens (`FungiStop‑X`).
- It cited sources in CJK brackets (`【G1】`), which broke source extraction.

**Prompt experiment** (all 6 cases, identical context for both prompts):

| Metric | v1 (naive) | v2 (grounded) | Δ |
|---|---|---|---|
| Faithfulness | 0.21 (0% pass) | **0.84** (83% pass) | **+0.63** |
| Answer Relevancy | 0.81 (83%) | **0.96** (100%) | **+0.14** |
| Key Facts | 1.00 | 1.00 | 0.00 |

- **Faithfulness:** v1's "use your agricultural knowledge" answers are mostly unsupported by the retrieved context. Per-case faithfulness ranges from 0.06 to 0.35.
- **Relevancy:** v1's broad answers also drift off-topic. The trends question scores 0.07 relevancy under v1.
- **Key facts:** both prompts name the right disease and product, because the context contains them. The difference is whether the rest of the answer can be trusted.
- **Variance:** LLM-judged scores vary between runs. For example, `early_blight_manisa` v2 faithfulness scored 1.00 in one run and 0.60 in another, so read the means rather than single cases.

### Free-tier budget and timing
Groq's free tier allows 8K tokens/min and 200K tokens/day **per model**. The judge and synthesis both use gpt-oss-120b.

Measured usage (gpt-oss-120b = synthesis + judge):

| Run | gpt-oss-120b tokens | gpt-oss-20b tokens | Wall time |
|---|---|---|---|
| `eval/test_suite.py` (6 cases) | ~48K | ~5.5K | ~4 min |
| `eval/prompt_experiment.py` (6 cases × 2 prompts) | ~119K | ~5.5K | ~11 min |

Most of the wall time is waiting on the per-minute limit; the client retries 429s automatically.

Each run fits in one day's budget; both together (~170K) come close to the 200K cap. On the free tier, run them on different days, or use `--limit`, or use Groq's pay-as-you-go Developer tier, where a full run costs a few cents. When Groq's daily cap is hit, the client raises a clear `LLMRateLimitError`, and the scripts record it instead of crashing mid-run. Each script ends with an LLM usage summary, showing calls, tokens and estimated cost per step.

Optional: set `CONFIDENT_API_KEY` in `.env` to upload DeepEval runs to Confident AI. Everything works locally without it, and telemetry is off via `DEEPEVAL_TELEMETRY_OPT_OUT=1`.

## Project layout

```
docker-compose.yml   neo4j + app services        Dockerfile      app image (CPU torch, model + LanceDB baked in)
.env.example         GROQ_API_KEY / PORTKEY_API_KEY are REQUIRED
requirements.txt     app + seeds                  requirements-dev.txt   + deepeval (host)
seed/                seed_neo4j.py (20 nodes), seed_lancedb.py (8 notes)
app/
  main.py            FastAPI: POST /query, GET /health (fails fast without Portkey)
  pipeline.py        orchestrator → retrieval → social listening → synthesis
  llm_client.py      Portkey-only LLM client: tracing, JSON mode, rate-limit handling, local usage mirror
  graph_client.py    Neo4j driver + region canonicalisation
  vector_client.py   MiniLM embeddings + LanceDB
  config.py          env/.env settings
  agents/            orchestrator.py, retrieval.py, social_listening.py, synthesis.py
eval/
  dataset.py         6 fixed cases with expected key facts
  metrics.py         Portkey/Groq DeepEval judge, metric set, KeyFacts metric, reporting helpers
  test_suite.py      structured suite (script + `deepeval test run`)
  prompt_experiment.py   v1 vs v2 synthesis prompt comparison
test_e2e.py          step-by-step end-to-end demo with Portkey traces
```

## Expected output of the example query

`python test_e2e.py` output from a real run on 2026-09-22 (lightly trimmed). The wording varies from run to run; the structure and facts stay the same.

```
QUERY: White spots showed up on my tomato leaves in İzmir

[1] ORCHESTRATOR AGENT (intent + entities + routing)
    intent=diagnose_problem  crop=tomato  region=İzmir
    symptoms=['white spots']  keywords=['white', 'spots']
    run_retrieval=True  run_social_listening=True

[2] RETRIEVAL AGENT (hybrid: Neo4j graph + LanceDB vectors)
    graph strategy: symptom_match
    [G1] (score 4) Tomato (grown in İzmir) is affected by Powdery Mildew: Fungal disease that
    develops in humid conditions. Symptoms: white spots. Treated by: FungiStop-X (fungicide).
    [D1] (cosine distance 0.2011) Powdery mildew and humidity: ...
    [D2] (cosine distance 0.3317) Managing powdery mildew: ...
    [D3] (cosine distance 0.5117) Aegean late-summer disease pressure: ...
    graph nodes: Crop:tomato, Region:İzmir, Disease:Powdery Mildew, Symptom:white spots, Product:FungiStop-X

[3] SOCIAL LISTENING AGENT (mock posts from the graph)
    region filter: İzmir  posts: 2  concern: medium
    summary: Growers in İzmir report tiny moths creating tunnels in greenhouse tomato leaves and
    white spots on tomatoes, possibly due to humidity.

[4] SYNTHESIS AGENT (prompt v2)
    **Likely cause**
    White spots on tomato leaves in İzmir are typical of powdery mildew, a fungal disease that
    appears as small white, powdery spots, especially under humid conditions[G1][D1].
    **What to do**
    1. Remove the most affected leaves to reduce inoculum[D2].
    2. Improve air circulation around plants to lower humidity[D2].
    3. Apply the registered fungicide FungiStop-X at the first sign of disease, following label
    rates and rotating modes of action to prevent resistance[G1][D2].
    **Prevention**
    - Scout regularly, especially in late summer when warm days and humid nights increase risk in
    the Aegean region[D3].
    - Maintain good spacing and pruning to enhance airflow[D2].
    **What other growers report**
    Recent İzmir growers have noted white spots on tomatoes and linked them to humidity, confirming
    powdery mildew concerns[S3].
    cited sources: G1, D1, D2, D3, S3

[PORTKEY TRACE] trace_id = agri-0a559ecf-e465-4de2-aace-804cc8ba63ce
    step             model                   latency  tokens     cost $  span_id           gateway trace id
    orchestrator     openai/gpt-oss-20b        877ms     528   0.000070  02468ffc1b9945b8  agri-0a559ecf-...
    social_listening openai/gpt-oss-20b        456ms     374   0.000045  ec6b5c39c9db4f76  agri-0a559ecf-...
    synthesis        openai/gpt-oss-120b      1013ms     964   0.000262  8c4fc1fcb51349fc  agri-0a559ecf-...
    total: 3 LLM calls, 2346 ms, 1866 tokens, $0.000376

[CHECKS]
    PASS  mentions Powdery Mildew
    PASS  mentions FungiStop-X
    PASS  includes the social trend
    PASS  Portkey echoed the trace_id
```

The "gateway trace id" column is the `x-portkey-trace-id` response header returned by Portkey. It matches the trace id we sent, which shows every call really went through the gateway. The same trace appears in the Portkey dashboard with one span per agent.
