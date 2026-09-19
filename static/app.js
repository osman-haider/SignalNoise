// Signal/Noise -- frontend logic. No framework, no build step: this is a
// small demo, and a plain fetch()-driven page is the fastest honest way to
// show the backend's behavior.

const API_BASE = ""; // same origin

const SUGGESTED_QUERIES = ["navy", "fleet", "strike", "carrier", "target", "mission"];

let currentMode = "naive";

// ---- DOM refs --------------------------------------------------------

const queryInput = document.getElementById("query-input");
const searchBtn = document.getElementById("search-btn");
const modeNaiveBtn = document.getElementById("mode-naive");
const modeHybridBtn = document.getElementById("mode-hybrid");
const suggestedQueriesEl = document.getElementById("suggested-queries");

const disambiguationPanel = document.getElementById("disambiguation-panel");
const senseRow = document.getElementById("sense-row");
const expansionTermsEl = document.getElementById("expansion-terms");
const degradedNotice = document.getElementById("degraded-notice");

const resultsSummary = document.getElementById("results-summary");
const resultsList = document.getElementById("results-list");

const runEvalBtn = document.getElementById("run-eval-btn");
const evalResultsEl = document.getElementById("eval-results");

// ---- Setup -------------------------------------------------------------

function init() {
  SUGGESTED_QUERIES.forEach((term) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = term;
    chip.addEventListener("click", () => {
      queryInput.value = term;
      runSearch();
    });
    suggestedQueriesEl.appendChild(chip);
  });

  modeNaiveBtn.addEventListener("click", () => setMode("naive"));
  modeHybridBtn.addEventListener("click", () => setMode("hybrid"));
  searchBtn.addEventListener("click", runSearch);
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runSearch();
  });
  runEvalBtn.addEventListener("click", runEvaluation);
}

function setMode(mode) {
  currentMode = mode;
  modeNaiveBtn.classList.toggle("active", mode === "naive");
  modeHybridBtn.classList.toggle("active", mode === "hybrid");
  if (queryInput.value.trim()) runSearch();
}

// ---- Search --------------------------------------------------------------

async function runSearch() {
  const query = queryInput.value.trim();
  if (!query) return;

  searchBtn.disabled = true;
  resultsSummary.innerHTML = '<span class="loading">Searching…</span>';
  resultsList.innerHTML = "";
  disambiguationPanel.classList.remove("visible");

  try {
    const res = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode: currentMode, k: 10 }),
    });
    if (!res.ok) throw new Error(`Search failed (${res.status})`);
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    resultsSummary.textContent = `Error: ${err.message}`;
  } finally {
    searchBtn.disabled = false;
  }
}

function renderResults(data) {
  const { mode, query, results, disambiguation, degraded, degraded_reason } = data;

  if (mode === "hybrid") {
    disambiguationPanel.classList.add("visible");
    renderDisambiguation(disambiguation, degraded, degraded_reason);
  }

  const nOffTopic = results.filter((r) => r.topic.endsWith("_other")).length;
  resultsSummary.textContent =
    `${results.length} results for "${query}" (${mode} mode) — ` +
    `${nOffTopic} of the top ${results.length} are off-topic for a military/OSINT context.`;

  resultsList.innerHTML = "";
  results.forEach((r, i) => {
    const isOffTopic = r.topic.endsWith("_other");
    const card = document.createElement("div");
    card.className = "result-card" + (isOffTopic ? " off-topic" : "");

    const scoreText =
      r.semantic_score !== null && r.semantic_score !== undefined
        ? `combined ${r.score.toFixed(3)} · bm25 ${r.bm25_score.toFixed(3)} · semantic ${r.semantic_score.toFixed(3)}`
        : `bm25 ${r.bm25_score.toFixed(3)}`;

    card.innerHTML = `
      <div class="result-rank">${i + 1}</div>
      <div class="result-body">
        <p class="result-text">${escapeHtml(r.text)}</p>
        <div class="result-meta">
          <span class="flag">${isOffTopic ? "OFF-TOPIC" : "ON-TOPIC"}</span>
          <span>${scoreText}</span>
        </div>
      </div>
    `;
    resultsList.appendChild(card);
  });
}

function renderDisambiguation(disambiguation, degraded, degradedReason) {
  senseRow.innerHTML = "";
  expansionTermsEl.textContent = "";
  degradedNotice.classList.add("hidden");

  if (disambiguation) {
    (disambiguation.senses || []).forEach((sense) => {
      const pill = document.createElement("span");
      pill.className = "sense-pill" + (sense === disambiguation.selected_sense ? " selected" : "");
      pill.textContent = sense;
      senseRow.appendChild(pill);
    });
    if (disambiguation.expansion_terms && disambiguation.expansion_terms.length) {
      expansionTermsEl.textContent = "expanded with: " + disambiguation.expansion_terms.join(", ");
    }
  }

  if (degraded && degradedReason) {
    degradedNotice.textContent = degradedReason;
    degradedNotice.classList.remove("hidden");
  }
}

// ---- Evaluation ------------------------------------------------------

async function runEvaluation() {
  runEvalBtn.disabled = true;
  evalResultsEl.innerHTML = '<p class="loading">Running naive and hybrid search across the full eval set…</p>';

  try {
    const res = await fetch(`${API_BASE}/api/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ k: 10 }),
    });
    if (!res.ok) throw new Error(`Evaluation failed (${res.status})`);
    const data = await res.json();
    renderEvaluation(data);
  } catch (err) {
    evalResultsEl.innerHTML = `<p>Error: ${escapeHtml(err.message)}</p>`;
  } finally {
    runEvalBtn.disabled = false;
  }
}

function renderEvaluation(data) {
  const { naive, hybrid, k } = data;

  const delta = (a, b) => {
    const d = b - a;
    const sign = d >= 0 ? "+" : "";
    return `<span class="metric-delta">${sign}${d.toFixed(3)}</span>`;
  };

  let html = `
    <table class="eval-table">
      <thead>
        <tr><th></th><th>Naive keyword</th><th>Hybrid + disambiguation</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Precision@${k}</td>
          <td class="metric">${naive.mean_precision.toFixed(3)}</td>
          <td class="metric">${hybrid.mean_precision.toFixed(3)} ${delta(naive.mean_precision, hybrid.mean_precision)}</td>
        </tr>
        <tr>
          <td>Recall@${k}</td>
          <td class="metric">${naive.mean_recall.toFixed(3)}</td>
          <td class="metric">${hybrid.mean_recall.toFixed(3)} ${delta(naive.mean_recall, hybrid.mean_recall)}</td>
        </tr>
        <tr>
          <td>NDCG@${k}</td>
          <td class="metric">${naive.mean_ndcg.toFixed(3)}</td>
          <td class="metric">${hybrid.mean_ndcg.toFixed(3)} ${delta(naive.mean_ndcg, hybrid.mean_ndcg)}</td>
        </tr>
      </tbody>
    </table>
  `;

  if (hybrid.degraded) {
    html += `<div class="degraded-notice">Hybrid mode ran in degraded (lexical-only) mode for at least one query during this evaluation run.</div>`;
  }

  html += `
    <div class="per-query-table">
      <table class="eval-table">
        <thead>
          <tr><th>Query</th><th>Kind</th><th>Naive P/R/NDCG</th><th>Hybrid P/R/NDCG</th></tr>
        </thead>
        <tbody>
          ${naive.per_query
            .map((row) => {
              const hybridRow = hybrid.per_query.find((r) => r.query_id === row.query_id);
              return `
                <tr>
                  <td class="metric">${escapeHtml(row.query_text)}</td>
                  <td>${row.kind}</td>
                  <td class="metric">${row.precision_at_k.toFixed(2)} / ${row.recall_at_k.toFixed(2)} / ${row.ndcg_at_k.toFixed(2)}</td>
                  <td class="metric">${hybridRow.precision_at_k.toFixed(2)} / ${hybridRow.recall_at_k.toFixed(2)} / ${hybridRow.ndcg_at_k.toFixed(2)}</td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;

  evalResultsEl.innerHTML = html;
}

// ---- Utility -----------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

init();
