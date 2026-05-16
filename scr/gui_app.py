from pathlib import Path
import sys
from functools import lru_cache
from time import perf_counter
import heapq

from flask import Flask, jsonify, request, render_template_string


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "scr" else CURRENT_DIR
NGRAM_DIR = PROJECT_ROOT / "scr" / "ngram"

sys.path.insert(0, str(NGRAM_DIR))

from ngram_model import NGramModel


app = Flask(__name__)

BEST_LAMBDAS = {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.9}

_NGRAM_PATHS = {
    "tinystories": PROJECT_ROOT / "models" / "ngram" / "Tiny_stories_ngram_model.pkl",
    "wikitext2":   PROJECT_ROOT / "models" / "ngram" / "wikitext2_ngram_model.pkl",
}

NGRAM_MODELS = {}
for _key, _path in _NGRAM_PATHS.items():
    if _path.exists():
        _m = NGramModel.load(_path)
        if not hasattr(_m, "prefix_index") or not _m.prefix_index:
            _m._build_prefix_index()
        NGRAM_MODELS[_key] = _m
        print(f"Loaded n-gram model [{_key}]: {_path.name}")
    else:
        print(f"Warning: model not found for [{_key}]: {_path}")

if not NGRAM_MODELS:
    raise RuntimeError("No n-gram models found. Train them first.")

DEFAULT_DATASET = "tinystories" if "tinystories" in NGRAM_MODELS else next(iter(NGRAM_MODELS))


def fast_predict_interpolated(context, prefix, model, top_k=5, lambdas=BEST_LAMBDAS):
    candidates = model.get_candidates(prefix)
    scored = []
    for word in candidates:
        score = model.interpolated_probability(context=context, word=word, lambdas=lambdas)
        scored.append((word, score))
    ranked = heapq.nsmallest(
        top_k,
        scored,
        key=lambda item: (
            -item[1],
            -model.word_counts.get(item[0], 0),
            len(item[0]),
            item[0],
        ),
    )
    return ranked, len(candidates)


@lru_cache(maxsize=4096)
def cached_suggest(text, top_k, dataset_key):
    model = NGRAM_MODELS[dataset_key]
    context, prefix = model.parse_text_input(text)
    suggestions, candidate_count = fast_predict_interpolated(
        context=context, prefix=prefix, model=model, top_k=top_k
    )
    return context, prefix, suggestions, candidate_count


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Word Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

<style>
    :root {
        --bg: #080a12;
        --panel: rgba(17, 24, 39, 0.78);
        --panel-strong: rgba(24, 31, 46, 0.94);
        --text: #f8fafc;
        --muted: #94a3b8;
        --soft: #cbd5e1;
        --border: rgba(255, 255, 255, 0.12);
        --accent: #8b5cf6;
        --accent-2: #22d3ee;
        --good: #34d399;
        --shadow: 0 28px 90px rgba(0, 0, 0, 0.42);
    }

    * { box-sizing: border-box; }

    html { min-height: 100%; color-scheme: dark; }

    body {
        min-height: 100vh;
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        color: var(--text);
        background:
            radial-gradient(circle at 12% 8%, rgba(139, 92, 246, 0.32), transparent 30%),
            radial-gradient(circle at 88% 14%, rgba(34, 211, 238, 0.2), transparent 32%),
            radial-gradient(circle at 50% 100%, rgba(52, 211, 153, 0.09), transparent 30%),
            linear-gradient(135deg, #070913 0%, #111827 52%, #070913 100%);
    }

    body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: linear-gradient(to bottom, black, transparent 80%);
    }

    .page {
        width: min(1120px, calc(100% - 36px));
        margin: 0 auto;
        padding: 46px 0;
    }

    .hero { margin-bottom: 28px; }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 14px;
        padding: 7px 11px;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.055);
        color: #dbeafe;
        font-size: 13px;
        backdrop-filter: blur(14px);
    }

    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--good);
        box-shadow: 0 0 18px rgba(52, 211, 153, 0.8);
    }

    h1 {
        margin: 0;
        font-size: clamp(38px, 7vw, 72px);
        letter-spacing: -0.07em;
        line-height: 0.95;
        background: linear-gradient(135deg, #ffffff, #c4b5fd 48%, #67e8f9);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .subtitle {
        max-width: 660px;
        margin: 16px 0 0;
        color: var(--muted);
        font-size: 18px;
        line-height: 1.6;
    }

    .workspace {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(310px, 0.75fr);
        gap: 22px;
        align-items: start;
    }

    .card {
        border: 1px solid var(--border);
        border-radius: 28px;
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.035)),
            var(--panel);
        box-shadow: var(--shadow);
        backdrop-filter: blur(18px);
    }

    .composer { padding: 22px; }

    .suggestion-panel {
        padding: 18px;
        position: sticky;
        top: 22px;
    }

    .section-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        margin-bottom: 14px;
    }

    .section-title {
        margin: 0;
        font-size: 15px;
        color: #e2e8f0;
        letter-spacing: 0.02em;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 10px;
        border: 1px solid rgba(255, 255, 255, 0.11);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.055);
        color: #bfdbfe;
        font-size: 12px;
        white-space: nowrap;
    }

    /* ── selectors ── */
    .selectors {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-bottom: 14px;
        padding: 14px 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.028);
    }

    .selector-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .selector-label {
        min-width: 52px;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        flex-shrink: 0;
    }

    .toggle-group {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }

    .toggle-btn {
        padding: 5px 13px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.05);
        color: var(--muted);
        font-size: 13px;
        font-family: inherit;
        cursor: pointer;
        transition: all 0.15s ease;
        white-space: nowrap;
    }

    .toggle-btn:hover:not(:disabled) {
        border-color: rgba(34, 211, 238, 0.4);
        color: #e2e8f0;
    }

    .toggle-btn.active {
        border-color: rgba(139, 92, 246, 0.7);
        background: rgba(139, 92, 246, 0.18);
        color: #e9d5ff;
        box-shadow: 0 0 14px rgba(139, 92, 246, 0.2);
    }

    .toggle-btn:disabled {
        opacity: 0.38;
        cursor: not-allowed;
    }

    textarea {
        width: 100%;
        min-height: 240px;
        resize: vertical;
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 22px;
        padding: 20px 22px;
        background: rgba(7, 11, 21, 0.84);
        color: var(--text);
        font: inherit;
        font-size: 23px;
        line-height: 1.55;
        caret-color: var(--accent-2);
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.05),
            0 20px 55px rgba(0, 0, 0, 0.22);
        transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }

    textarea::placeholder { color: #64748b; }

    textarea:focus {
        outline: none;
        border-color: rgba(34, 211, 238, 0.72);
        background: rgba(8, 13, 25, 0.96);
        box-shadow:
            0 0 0 4px rgba(34, 211, 238, 0.12),
            0 22px 70px rgba(0, 0, 0, 0.36),
            0 0 48px rgba(139, 92, 246, 0.16);
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 16px;
    }

    .info-box {
        min-height: 76px;
        padding: 13px 14px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.045);
    }

    .info-label {
        display: block;
        margin-bottom: 7px;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .info-value {
        color: #f8fafc;
        font-size: 14px;
        line-height: 1.45;
        word-break: break-word;
    }

    .suggestions-list {
        display: flex;
        flex-direction: column;
        gap: 10px;
        min-height: 280px;
    }

    .suggestion-button {
        width: 100%;
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr) auto;
        align-items: center;
        gap: 12px;
        padding: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        background:
            linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(34, 211, 238, 0.09)),
            rgba(255, 255, 255, 0.055);
        color: white;
        cursor: pointer;
        text-align: left;
        transition: transform 0.14s ease, border-color 0.14s ease, background 0.14s ease, box-shadow 0.14s ease;
    }

    .suggestion-button:hover {
        transform: translateY(-2px);
        border-color: rgba(34, 211, 238, 0.45);
        background:
            linear-gradient(135deg, rgba(139, 92, 246, 0.32), rgba(34, 211, 238, 0.16)),
            rgba(255, 255, 255, 0.075);
        box-shadow: 0 16px 38px rgba(34, 211, 238, 0.12);
    }

    .suggestion-button:active { transform: translateY(0); }

    .rank {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.09);
        color: #bae6fd;
        font-weight: 800;
        font-size: 13px;
    }

    .word {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 20px;
        font-weight: 750;
        letter-spacing: -0.02em;
    }

    .score {
        padding: 5px 8px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        color: var(--soft);
        font-size: 12px;
        font-variant-numeric: tabular-nums;
    }

    .empty {
        padding: 16px;
        border: 1px dashed rgba(255, 255, 255, 0.16);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        line-height: 1.5;
    }

    .hint {
        margin: 16px 0 0;
        padding-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.09);
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
    }

    kbd {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 24px;
        padding: 2px 7px;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-bottom-color: rgba(255, 255, 255, 0.32);
        border-radius: 7px;
        background: rgba(255, 255, 255, 0.075);
        color: #e0f2fe;
        font-size: 12px;
        box-shadow: inset 0 -1px 0 rgba(255, 255, 255, 0.08);
    }

    ::selection { background: rgba(34, 211, 238, 0.35); }

    @media (max-width: 860px) {
        .workspace { grid-template-columns: 1fr; }
        .suggestion-panel { position: static; }
        .info-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 560px) {
        .page { width: min(100% - 24px, 1120px); padding: 28px 0; }
        .composer, .suggestion-panel { padding: 16px; border-radius: 22px; }
        textarea { min-height: 190px; font-size: 19px; border-radius: 18px; }
        .suggestion-button { grid-template-columns: 36px minmax(0, 1fr); }
        .score { grid-column: 2; width: fit-content; }
    }
</style>
</head>

<body>
    <div class="page">
        <header class="hero">
            <div class="eyebrow">
                <span class="dot"></span>
                <span id="eyebrowText">N-gram · TinyStories</span>
            </div>

            <h1>Word Predictor</h1>

            <p class="subtitle">
                Type a sentence and get ranked next-word or prefix-completion suggestions from your trained language model.
            </p>
        </header>

        <main class="workspace">
            <section class="card composer">
                <div class="section-top">
                    <h2 class="section-title">Input text</h2>
                    <span class="pill" id="modeText">next-word prediction</span>
                </div>

                <div class="selectors">
                    <div class="selector-row">
                        <span class="selector-label">Model</span>
                        <div class="toggle-group" id="modelGroup">
                            <button class="toggle-btn active" data-model="ngram">N-gram</button>
                            <button class="toggle-btn" data-model="transformer" disabled>Transformer</button>
                        </div>
                    </div>
                    <div class="selector-row">
                        <span class="selector-label">Dataset</span>
                        <div class="toggle-group" id="datasetGroup">
                            <button class="toggle-btn active" data-dataset="tinystories">TinyStories</button>
                            <button class="toggle-btn" data-dataset="wikitext2">WikiText-2</button>
                        </div>
                    </div>
                </div>

                <textarea id="inputBox" placeholder="Start typing here..."></textarea>

                <div class="info-grid">
                    <div class="info-box">
                        <span class="info-label">Context</span>
                        <span class="info-value" id="contextText">[]</span>
                    </div>

                    <div class="info-box">
                        <span class="info-label">Current prefix</span>
                        <span class="info-value" id="prefixText">(empty)</span>
                    </div>

                    <div class="info-box">
                        <span class="info-label">Model status</span>
                        <span class="info-value" id="statusText">Ready.</span>
                    </div>
                </div>
            </section>

            <aside class="card suggestion-panel">
                <div class="section-top">
                    <h2 class="section-title">Suggestions</h2>
                    <span class="pill">Top matches</span>
                </div>

                <div id="suggestions" class="suggestions-list"></div>

                <p class="hint">
                    Click a suggestion to insert it into the text box.
                    Suggestions are shown vertically so the ranking is easier to scan.
                </p>
            </aside>
        </main>
    </div>

    <script>
        const inputBox       = document.getElementById("inputBox");
        const suggestionsDiv = document.getElementById("suggestions");
        const contextText    = document.getElementById("contextText");
        const prefixText     = document.getElementById("prefixText");
        const modeText       = document.getElementById("modeText");
        const statusText     = document.getElementById("statusText");
        const eyebrowText    = document.getElementById("eyebrowText");

        let selectedModel   = "ngram";
        let selectedDataset = "tinystories";

        let debounceTimer = null;
        let currentAbortController = null;
        let latestRequestId = 0;

        const MODEL_LABELS   = { ngram: "N-gram", transformer: "Transformer" };
        const DATASET_LABELS = { tinystories: "TinyStories", wikitext2: "WikiText-2" };

        function updateEyebrow() {
            eyebrowText.textContent =
                MODEL_LABELS[selectedModel] + " · " + DATASET_LABELS[selectedDataset];
        }

        function activateToggle(group, attr, value) {
            group.querySelectorAll(".toggle-btn").forEach(btn => {
                btn.classList.toggle("active", btn.dataset[attr] === value);
            });
        }

        document.getElementById("modelGroup").addEventListener("click", e => {
            const btn = e.target.closest(".toggle-btn");
            if (!btn || btn.disabled) return;
            selectedModel = btn.dataset.model;
            activateToggle(document.getElementById("modelGroup"), "model", selectedModel);
            updateEyebrow();
            scheduleFetchSuggestions();
        });

        document.getElementById("datasetGroup").addEventListener("click", e => {
            const btn = e.target.closest(".toggle-btn");
            if (!btn || btn.disabled) return;
            selectedDataset = btn.dataset.dataset;
            activateToggle(document.getElementById("datasetGroup"), "dataset", selectedDataset);
            updateEyebrow();
            scheduleFetchSuggestions();
        });

        function parseTextClientSide(text) {
            const lowerText = text.toLowerCase();

            if (lowerText.endsWith(" ")) {
                const trimmed = lowerText.trim();
                const words = trimmed.length === 0 ? [] : trimmed.split(/\\s+/);
                return { context: words, prefix: "" };
            }

            const trimmed = lowerText.trim();
            if (trimmed.length === 0) return { context: [], prefix: "" };

            const words = trimmed.split(/\\s+/);
            return { context: words.slice(0, -1), prefix: words[words.length - 1] };
        }

        function updateContextPrefixInstantly() {
            const parsed = parseTextClientSide(inputBox.value);
            contextText.textContent = JSON.stringify(parsed.context);
            prefixText.textContent  = parsed.prefix === "" ? "(empty)" : parsed.prefix;
            modeText.textContent    = parsed.prefix === "" ? "next-word prediction" : "prefix completion";
            return parsed;
        }

        function clearSuggestions() { suggestionsDiv.innerHTML = ""; }

        function renderSuggestions(data) {
            suggestionsDiv.innerHTML = "";

            if (!data.suggestions || data.suggestions.length === 0) {
                suggestionsDiv.innerHTML = "<div class='empty'>No suggestions found yet. Try typing another prefix or a longer context.</div>";
                return;
            }

            data.suggestions.forEach((item, index) => {
                const button = document.createElement("button");
                button.className = "suggestion-button";
                button.type = "button";

                const rankSpan  = document.createElement("span");
                rankSpan.className = "rank";
                rankSpan.textContent = String(index + 1).padStart(2, "0");

                const wordSpan  = document.createElement("span");
                wordSpan.className = "word";
                wordSpan.textContent = item.word;

                const scoreSpan = document.createElement("span");
                scoreSpan.className = "score";
                scoreSpan.textContent = Number(item.score).toExponential(2);

                button.appendChild(rankSpan);
                button.appendChild(wordSpan);
                button.appendChild(scoreSpan);
                button.addEventListener("click", () => insertSuggestion(item.word));
                suggestionsDiv.appendChild(button);
            });
        }

        async function fetchSuggestionsNow() {
            const text = inputBox.value;
            const requestId = ++latestRequestId;

            if (currentAbortController !== null) currentAbortController.abort();
            currentAbortController = new AbortController();

            statusText.textContent = "Updating...";

            try {
                const response = await fetch("/suggest", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        text:    text,
                        top_k:   7,
                        model:   selectedModel,
                        dataset: selectedDataset,
                    }),
                    signal: currentAbortController.signal,
                });

                if (!response.ok) throw new Error("Request failed.");

                const data = await response.json();
                if (requestId !== latestRequestId) return;

                contextText.textContent = JSON.stringify(data.context);
                prefixText.textContent  = data.prefix === "" ? "(empty)" : data.prefix;
                modeText.textContent    = data.mode;

                renderSuggestions(data);
                statusText.textContent =
                    `${data.elapsed_ms.toFixed(1)} ms · ${data.candidate_count} candidates`;

            } catch (error) {
                if (error.name === "AbortError") return;
                statusText.textContent = "Could not update.";
            }
        }

        function scheduleFetchSuggestions() {
            updateContextPrefixInstantly();
            if (debounceTimer !== null) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(fetchSuggestionsNow, 70);
        }

        function insertSuggestion(word) {
            const text   = inputBox.value;
            const parsed = parseTextClientSide(text);
            let newText  = text;

            if (parsed.prefix.length > 0) newText = text.slice(0, text.length - parsed.prefix.length);
            if (newText.length > 0 && !newText.endsWith(" ")) newText += " ";

            inputBox.value = newText + word + " ";
            inputBox.focus();
            updateContextPrefixInstantly();
            clearSuggestions();
            if (debounceTimer !== null) clearTimeout(debounceTimer);
            fetchSuggestionsNow();
        }

        inputBox.addEventListener("input", scheduleFetchSuggestions);

        updateEyebrow();
        updateContextPrefixInstantly();
        fetchSuggestionsNow();
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/suggest", methods=["POST"])
def suggest():
    start_time = perf_counter()

    data = request.get_json() or {}
    text        = data.get("text", "")
    top_k       = max(1, min(int(data.get("top_k", 5)), 10))
    dataset_key = data.get("dataset", DEFAULT_DATASET)

    if dataset_key not in NGRAM_MODELS:
        dataset_key = DEFAULT_DATASET

    context, prefix, suggestions, candidate_count = cached_suggest(text, top_k, dataset_key)
    elapsed_ms = (perf_counter() - start_time) * 1000

    return jsonify({
        "context":         context,
        "prefix":          prefix,
        "mode":            "prefix completion" if prefix else "next-word prediction",
        "candidate_count": candidate_count,
        "elapsed_ms":      elapsed_ms,
        "dataset":         dataset_key,
        "suggestions":     [{"word": word, "score": float(score)} for word, score in suggestions],
    })


if __name__ == "__main__":
    print(f"Available models: {list(NGRAM_MODELS.keys())}")
    app.run(debug=True, use_reloader=False, threaded=True)
