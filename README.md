# Signal/Noise

A small hybrid-search prototype built as a portfolio demo for a Senior
Software Engineer (Search) role. It compares **naive keyword search** against
**hybrid search with LLM query disambiguation**, on the same corpus, and
proves the difference with a real Precision/Recall/NDCG evaluation harness
— rather than just eyeballing one example query.

## Why this exists

Search on ambiguous single-word queries (e.g. `navy`) is a documented,
independently-observed failure mode in OSINT-style keyword search: plain
keyword matching can't tell "naval fleet" from "anime fandom content that
happens to contain the same word." This project reproduces that exact
problem on a small, fully synthetic corpus, fixes it with a proportionate
combination of BM25 + LLM-based query disambiguation + embedding re-ranking,
and measures whether the fix actually worked.

**Every document in `data/corpus.json` is self-authored for this project.**
Nothing is scraped from real news, and no real company's data, customers, or
product is represented here. See the accompanying research documents for
the public evidence this idea is based on.

## Project layout

```
signal-noise/
├── app/
│   ├── bm25.py              # dependency-free BM25 implementation
│   ├── config.py            # reads .env -- the ONLY place that touches os.environ
│   ├── data.py               # loads corpus.json / eval_queries.json
│   ├── embeddings.py         # OpenAI embeddings + disk cache + cosine similarity
│   ├── llm_disambiguate.py   # OpenAI chat completion -> structured disambiguation JSON
│   ├── search_engine.py      # naive vs hybrid search, incl. graceful degradation
│   ├── evaluation.py         # Precision@k / Recall@k / NDCG@k, from scratch
│   └── main.py               # FastAPI app + routes, serves the frontend
├── static/                   # plain HTML/CSS/JS frontend, no build step
├── data/                     # the synthetic corpus + curated eval query set
├── scripts/
│   ├── generate_corpus.py    # (re)generates data/corpus.json + eval_queries.json
│   └── build_embeddings.py   # precomputes + caches corpus embeddings (run once)
├── cache/                    # corpus_embeddings.npy gets written here (gitignored)
├── requirements.txt
├── .env.example
└── Dockerfile
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set OFFICIAL_OPENAI_API_KEY (and OFFICIAL_OPENAI_BASE_URL / OFFICIAL_OPENAI_MODEL_NAME
# if you're not using the default OpenAI endpoint/model)
```

The corpus and eval query set are already generated and committed
(`data/corpus.json`, `data/eval_queries.json`). If you want to regenerate
them (e.g. after editing `scripts/generate_corpus.py`):

```bash
python scripts/generate_corpus.py
```

Then precompute corpus embeddings **once**, offline (this is the only part
of the pipeline that embeds all ~100 documents; live search only ever
embeds the single query):

```bash
python scripts/build_embeddings.py
```

Run the server:

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 — the frontend is served directly from `/static`.

## What "hybrid mode" actually does

1. The query is sent to the configured chat model with a fixed "analyst
   context" string, asking it to identify possible senses of the query term
   and pick the one matching that context (see `app/llm_disambiguate.py`).
2. The query is expanded with the model's suggested expansion terms, and
   BM25 runs against that expanded query.
3. The expanded query is embedded and compared (cosine similarity) against
   the precomputed corpus embeddings.
4. BM25 and semantic scores are min-max normalized and combined with a
   weighted sum (`HYBRID_BM25_WEIGHT` / `HYBRID_SEMANTIC_WEIGHT` in `.env`,
   default 0.4 / 0.6).

If step 1 or step 3 fails for any reason (missing API key, network error,
malformed response), the request **does not error out** — it degrades to
lexical-only (BM25) results and the API response includes
`"degraded": true` and a human-readable `degraded_reason`, which the
frontend surfaces directly rather than hiding the failure.

## Evaluation harness

`POST /api/evaluate` runs every query in `data/eval_queries.json` — six
deliberately ambiguous single-word queries plus four "control" queries that
are already unambiguous — through both search modes, and reports mean
Precision@k / Recall@k / NDCG@k for each, plus a per-query breakdown. This
is the same metric vocabulary named in the job posting this project is
built around.

**A note on the numbers you'll actually see:** during development, this
logic was validated end-to-end using a crude, keyword-overlap-based fake
embedding function (since real API access wasn't available in the dev
sandbox) — that test showed an aggregate improvement (precision ~0.45 →
~0.54, NDCG ~0.58 → ~0.69) but a mixed picture on a couple of individual
queries, which is realistic: no single technique wins on every query, which
is exactly why an averaged evaluation harness matters more than eyeballing
one result list. With real OpenAI embeddings, expect a larger and more
consistent improvement, since real embeddings separate "naval fleet" from
"anime fandom" far better than a 2-dimensional keyword-count heuristic can.
Re-run `scripts/build_embeddings.py` with a real API key and then
`POST /api/evaluate` to see your own numbers.

## Known limitations (worth being upfront about)

- The BM25 tokenizer does no stemming/lemmatization, so exact word-form
  mismatches (e.g. query expansion term "destroyers" vs. a document
  containing only "destroyer") can cause misses. A production system would
  add stemming or a lemma-aware tokenizer.
- The corpus is ~100 documents across 6 ambiguous terms — enough to
  demonstrate and measure the effect, not a claim about performance at real
  corpus scale.
- Binary-ish graded relevance (0 or 2) is used for NDCG for simplicity; a
  production eval set would likely use finer-grained human judgments.
- This project is not a claim that any specific company's search product
  has this weakness — see the accompanying research documents for what is
  and isn't publicly evidenced.

## Deployment

```bash
docker build -t signal-noise .
docker run -p 8000:8000 --env-file .env signal-noise
```

Then run `python scripts/build_embeddings.py` once against the running
container's environment (or bake a pre-built `cache/corpus_embeddings.npy`
into the image) — see the comment in `Dockerfile` for why this is a
deliberate runtime step rather than a build-time one.

### Vercel

Vercel's Python runtime gives the deployed function a **read-only**
filesystem (only `/tmp` is writable, and it's ephemeral per instance) and
has no arbitrary build step whose output gets reliably attached back to the
function bundle. That means `scripts/generate_corpus.py` and
`scripts/build_embeddings.py` can't run as part of the Vercel build/start
the way they can with Docker.

Because the corpus is small and fixed, the fix is to generate everything
**locally** and commit the results, so they ship as static files inside the
deployment bundle (Vercel bundles every file in the repo for Python
functions, not just imported ones):

```bash
python scripts/generate_corpus.py
python scripts/build_embeddings.py
git add data/corpus.json data/eval_queries.json cache/corpus_embeddings.npy cache/corpus_embeddings.meta.json
git commit -m "Regenerate corpus and embeddings cache"
git push
```

`app/search_engine.py` only ever *reads* `cache/corpus_embeddings.npy` at
request time — it never tries to (re)build it — so this is safe on a
read-only filesystem. Whenever `data/corpus.json` changes, rerun both
scripts and commit the updated cache alongside it; if the cache doesn't
match the current corpus hash, the app degrades to lexical-only search
instead of erroring, but the fix is always "regenerate and recommit," not a
runtime rebuild.
