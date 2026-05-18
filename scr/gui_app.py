from pathlib import Path
import sys
from functools import lru_cache
from time import perf_counter
import heapq

from flask import Flask, jsonify, request, render_template_string

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "scr" else CURRENT_DIR
NGRAM_DIR        = PROJECT_ROOT / "scr" / "ngram"
TRANSFORMER_DIR  = PROJECT_ROOT / "scr" / "transformer"

sys.path.insert(0, str(NGRAM_DIR))
sys.path.insert(0, str(TRANSFORMER_DIR))
sys.path.insert(0, str(CURRENT_DIR))

from ngram_model import NGramModel
from spell_corrector import SpellCorrector

TRANSFORMER_AVAILABLE = False
try:
    from transformer_model import TransformerPredictor
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    print(f"Transformer not available: {e}")

app = Flask(__name__)

BEST_LAMBDAS = {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.9}

_NGRAM_PATHS = {
    "tinystories": PROJECT_ROOT / "models" / "ngram" / "Tiny_stories_ngram_model.pkl",
    "wikitext2":   PROJECT_ROOT / "models" / "ngram" / "wikitext2_ngram_model.pkl",
}

_TRANSFORMER_PATHS = {
    "tinystories": {
        "checkpoint": PROJECT_ROOT / "models" / "transformer" / "tinystories" / "checkpoint_final.pt",
        "tokenizer":  PROJECT_ROOT / "models" / "transformer" / "tinystories" / "tokenizer.json",
        "ngram_pkl":  PROJECT_ROOT / "models" / "ngram" / "Tiny_stories_ngram_model.pkl",
        "train_text": PROJECT_ROOT / "scr" / "data" / "tiny_stories_transformer" / "tinystories_transformer_train.txt",
    },
    "wikitext2": {
        "checkpoint": PROJECT_ROOT / "models" / "transformer" / "wikitext2" / "checkpoint_final.pt",
        "tokenizer":  PROJECT_ROOT / "models" / "transformer" / "wikitext2" / "tokenizer.json",
        "ngram_pkl":  PROJECT_ROOT / "models" / "ngram" / "wikitext2_ngram_model.pkl",
        "train_text": PROJECT_ROOT / "scr" / "data" / "wikitext_2_transformer" / "wikitext2_transformer_train.txt",
    },
}

PREDICTORS = {"ngram": {}, "transformer": {}}

for _key, _path in _NGRAM_PATHS.items():
    if _path.exists():
        _m = NGramModel.load(_path)
        if not hasattr(_m, "prefix_index") or not _m.prefix_index:
            _m._build_prefix_index()
        PREDICTORS["ngram"][_key] = _m
        print(f"Loaded n-gram [{_key}]")
    else:
        print(f"N-gram not found [{_key}]")

if TRANSFORMER_AVAILABLE:
    for _key, _paths in _TRANSFORMER_PATHS.items():
        if _paths["checkpoint"].exists() and _paths["tokenizer"].exists():
            try:
                _tp = TransformerPredictor(
                    str(_paths["checkpoint"]), str(_paths["tokenizer"]), device="cpu"
                )
                if _paths["ngram_pkl"].exists():
                    _tp.load_word_vocab_from_ngram(str(_paths["ngram_pkl"]))
                elif _paths["train_text"].exists():
                    _tp.build_word_vocab_from_text(str(_paths["train_text"]))
                PREDICTORS["transformer"][_key] = _tp
                print(f"Loaded transformer [{_key}]")
            except Exception as e:
                print(f"Transformer load failed [{_key}]: {e}")
        else:
            print(f"Transformer checkpoint not found [{_key}]")

AVAILABLE = {
    "ngram":       list(PREDICTORS["ngram"].keys()),
    "transformer": list(PREDICTORS["transformer"].keys()),
}

if not AVAILABLE["ngram"] and not AVAILABLE["transformer"]:
    raise RuntimeError("No models found. Train them first.")

DEFAULT_MODEL   = "ngram" if AVAILABLE["ngram"] else "transformer"
DEFAULT_DATASET = (AVAILABLE[DEFAULT_MODEL] + ["tinystories"])[0]

_SPELL_CORRECTORS = {}

def get_spell_corrector(model_key, dataset_key, weight_mode, max_edit_dist):
    k = (model_key, dataset_key, weight_mode, max_edit_dist)
    if k not in _SPELL_CORRECTORS:
        _SPELL_CORRECTORS[k] = SpellCorrector(
            PREDICTORS[model_key][dataset_key],
            max_edit_dist=max_edit_dist, lambda_edit=0.5, mu_freq=0.1,
            weight_mode=weight_mode,
        )
    return _SPELL_CORRECTORS[k]


def fast_predict_ngram(context, prefix, model, top_k=5):
    candidates = model.get_candidates(prefix)
    scored = [
        (word, model.interpolated_probability(context=context, word=word, lambdas=BEST_LAMBDAS))
        for word in candidates
    ]
    ranked = heapq.nsmallest(
        top_k, scored,
        key=lambda item: (-item[1], -model.word_counts.get(item[0], 0), len(item[0]), item[0]),
    )
    return ranked, len(candidates)


@lru_cache(maxsize=4096)
def cached_suggest(text, top_k, model_key, dataset_key, spell_on, weight_mode, max_edit_dist, use_lm):
    predictor = PREDICTORS[model_key][dataset_key]
    context, prefix = predictor.parse_text_input(text)

    if spell_on:
        corrector = get_spell_corrector(model_key, dataset_key, weight_mode, max_edit_dist)
        suggestions = corrector.predict(context, prefix=prefix, top_k=top_k, include_scores=True, use_lm=use_lm)
        candidate_count = (
            len(corrector.get_spell_candidates(prefix)) if prefix else len(corrector.word_vocab)
        )
    elif model_key == "ngram":
        suggestions, candidate_count = fast_predict_ngram(context, prefix, predictor, top_k)
    else:
        suggestions = predictor.predict(context, prefix=prefix, top_k=top_k, include_scores=True)
        candidate_count = len(predictor.get_candidates(prefix))

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
        --spell: #f59e0b;
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
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--good);
        box-shadow: 0 0 18px rgba(52, 211, 153, 0.8);
        transition: background 0.3s, box-shadow 0.3s;
    }

    .dot.spell-active {
        background: var(--spell);
        box-shadow: 0 0 18px rgba(245, 158, 11, 0.8);
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

    .section-title { margin: 0; font-size: 15px; color: #e2e8f0; letter-spacing: 0.02em; }

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
        transition: background 0.2s, border-color 0.2s, color 0.2s;
    }

    .pill.spell-mode {
        border-color: rgba(245, 158, 11, 0.45);
        background: rgba(245, 158, 11, 0.12);
        color: #fde68a;
    }

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

    .selector-row.sub-row {
        padding-left: 64px;
        display: none;
    }

    .selector-row.sub-row.visible { display: flex; }

    .selector-desc {
        margin: 7px 0 0;
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.65;
        max-width: 480px;
    }

    .strategy-note {
        margin: 10px 0 0;
        padding: 10px 14px;
        border-left: 3px solid rgba(245, 158, 11, 0.5);
        background: rgba(245, 158, 11, 0.07);
        border-radius: 0 8px 8px 0;
        color: #cbd5e1;
        font-size: 13px;
        line-height: 1.65;
        display: none;
    }

    .strategy-note.visible { display: block; }

    .selector-label {
        min-width: 52px;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        flex-shrink: 0;
    }

    .toggle-group { display: flex; gap: 6px; flex-wrap: wrap; }

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

    .toggle-btn.active.spell-btn {
        border-color: rgba(245, 158, 11, 0.7);
        background: rgba(245, 158, 11, 0.18);
        color: #fde68a;
        box-shadow: 0 0 14px rgba(245, 158, 11, 0.2);
    }

    .toggle-btn:disabled { opacity: 0.38; cursor: not-allowed; }

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
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 20px 55px rgba(0, 0, 0, 0.22);
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
        width: 34px; height: 34px;
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
        .selector-row.sub-row { padding-left: 0; }
    }
</style>
</head>

<body>
    <div class="page">
        <header class="hero">
            <div class="eyebrow">
                <span class="dot" id="statusDot"></span>
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
                        <div>
                            <div class="toggle-group" id="modelGroup">
                                <button class="toggle-btn" data-model="ngram">N-gram</button>
                                <button class="toggle-btn" data-model="transformer">Transformer</button>
                            </div>
                            <p class="selector-desc">Language model for word prediction. When spell is on, overridden by the LM row.</p>
                        </div>
                    </div>
                    <div class="selector-row">
                        <span class="selector-label">Dataset</span>
                        <div class="toggle-group" id="datasetGroup">
                            <button class="toggle-btn" data-dataset="tinystories">TinyStories</button>
                            <button class="toggle-btn" data-dataset="wikitext2">WikiText-2</button>
                        </div>
                    </div>
                    <div class="selector-row">
                        <span class="selector-label">Spell</span>
                        <div>
                            <div class="toggle-group" id="spellGroup">
                                <button class="toggle-btn active" data-spell="off">Off</button>
                                <button class="toggle-btn spell-btn" data-spell="on">On</button>
                            </div>
                            <p class="selector-desc">Spell correction <em>extends</em> word prediction — exact matches still appear, typo corrections are added on top.</p>
                        </div>
                    </div>

                    <div class="selector-row sub-row" id="strategyRow">
                        <span class="selector-label">Strategy</span>
                        <div>
                            <div class="toggle-group" id="strategyGroup">
                                <button class="toggle-btn" data-strategy="s1">S1 · Edit ≤ 1</button>
                                <button class="toggle-btn active" data-strategy="s2">S2 · Edit ≤ 2</button>
                                <button class="toggle-btn" data-strategy="s3">S3 · Weighted</button>
                            </div>
                            <p class="selector-desc" id="strategyDesc"></p>
                        </div>
                    </div>

                    <div class="selector-row sub-row" id="lmRow">
                        <span class="selector-label">LM</span>
                        <div>
                            <div class="toggle-group" id="lmGroup">
                                <button class="toggle-btn active" data-lm="none">No model</button>
                                <button class="toggle-btn" data-lm="ngram">N-gram</button>
                                <button class="toggle-btn" data-lm="transformer">Transformer</button>
                            </div>
                            <p class="selector-desc" id="lmDesc"></p>
                        </div>
                    </div>

                    <div class="selector-row sub-row" id="weightRow">
                        <span class="selector-label">Weighting</span>
                        <div>
                            <div class="toggle-group" id="weightGroup">
                                <button class="toggle-btn active" data-weight="operation">Operation</button>
                                <button class="toggle-btn" data-weight="keyboard">Keyboard</button>
                            </div>
                            <p class="selector-desc" id="weightDesc"></p>
                        </div>
                    </div>

                    <div class="strategy-note" id="strategyNote">
                        Active config: <span id="strategyNoteText"></span>
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
                    Suggestions update on every keystroke.
                </p>
            </aside>
        </main>
    </div>

<script>
    const AVAILABLE = {{ available | tojson }};

    const inputBox        = document.getElementById("inputBox");
    const suggestionsDiv  = document.getElementById("suggestions");
    const contextText     = document.getElementById("contextText");
    const prefixText      = document.getElementById("prefixText");
    const modeText        = document.getElementById("modeText");
    const statusText      = document.getElementById("statusText");
    const eyebrowText     = document.getElementById("eyebrowText");
    const statusDot       = document.getElementById("statusDot");
    const strategyRow      = document.getElementById("strategyRow");
    const lmRow            = document.getElementById("lmRow");
    const weightRow        = document.getElementById("weightRow");
    const strategyNote     = document.getElementById("strategyNote");
    const strategyNoteText = document.getElementById("strategyNoteText");

    let selectedModel    = AVAILABLE.ngram.length > 0 ? "ngram" : "transformer";
    let selectedDataset  = (AVAILABLE[selectedModel] || [])[0] || "tinystories";
    let selectedSpell    = false;
    let selectedStrategy = "s2";
    let selectedLmMode   = "none";
    let selectedWeight   = "operation";

    let debounceTimer = null;
    let currentAbortController = null;
    let latestRequestId = 0;

    const MODEL_LABELS   = { ngram: "N-gram", transformer: "Transformer" };
    const DATASET_LABELS = { tinystories: "TinyStories", wikitext2: "WikiText-2" };

    const STRATEGY_META = {
        s1: { max_edit_dist: 1, weight_mode: "uniform" },
        s2: { max_edit_dist: 2, weight_mode: "uniform" },
        s3: { max_edit_dist: 2, weight_mode: "operation" },
    };

    const STRATEGY_DESCS = {
        s1: "S1 · Edit ≤ 1 — only words reachable with exactly one insertion, deletion, substitution, or transposition. Strictest and fastest.",
        s2: "S2 · Edit ≤ 2 — up to two edits allowed. Catches the majority of realistic typos without producing too many false matches.",
        s3: "S3 · Weighted edit ≤ 2 — some operations are penalised less than others. Choose a weighting scheme in the row below.",
    };

    const LM_DESCS = {
        none:        "No model — candidates ranked by edit distance and word frequency only. No context used. Fastest option.",
        ngram:       "N-gram LM — n-gram context probability re-ranks candidates. Words likely given the preceding words score higher.",
        transformer: "Transformer LM — deep contextual re-ranking. Slowest but most context-aware.",
    };

    const WEIGHT_DESCS = {
        operation: "Operation weights — transpositions (swapped adjacent letters, e.g. \"teh\" → \"the\") cost 0.5 instead of 1.0, since adjacent-letter swaps are the most common typing error.",
        keyboard:  "Keyboard weights — keys physically close on QWERTY (e.g. \"s\" ↔ \"a\", \"r\" ↔ \"t\") also cost 0.5, modelling fat-finger errors.",
    };

    function getSpellParams() {
        if (!selectedSpell) return { max_edit_dist: 2, weight_mode: "uniform", use_lm: false, model: selectedModel };
        const meta = STRATEGY_META[selectedStrategy];
        const wm = selectedStrategy === "s3" ? selectedWeight : meta.weight_mode;
        const useLm = selectedLmMode !== "none";
        const model = useLm ? selectedLmMode : selectedModel;
        return { max_edit_dist: meta.max_edit_dist, weight_mode: wm, use_lm: useLm, model };
    }

    function updateDescriptions() {
        const sDesc = document.getElementById("strategyDesc");
        const lDesc = document.getElementById("lmDesc");
        const wDesc = document.getElementById("weightDesc");
        if (sDesc) sDesc.textContent = STRATEGY_DESCS[selectedStrategy] || "";
        if (lDesc) lDesc.textContent = LM_DESCS[selectedLmMode] || "";
        if (wDesc) wDesc.textContent = WEIGHT_DESCS[selectedWeight] || "";
    }

    function updateStrategyNote() {
        if (!selectedSpell) { strategyNote.classList.remove("visible"); return; }
        const p = getSpellParams();
        const sNum = selectedStrategy.toUpperCase();
        const weightLabel = p.weight_mode === "uniform" ? "uniform" :
                            p.weight_mode === "operation" ? "operation-weighted" : "keyboard-weighted";
        const lmLabel = p.use_lm ? MODEL_LABELS[p.model] + " LM" : "no LM";
        strategyNoteText.textContent = `${sNum} · edit ≤ ${p.max_edit_dist} · ${weightLabel} · ${lmLabel}`;
        strategyNote.classList.add("visible");
    }

    function updateEyebrow() {
        const p = getSpellParams();
        const modelLabel = MODEL_LABELS[p.model] || MODEL_LABELS[selectedModel];
        let text = modelLabel + " · " + DATASET_LABELS[selectedDataset];
        if (selectedSpell) text += " · Spell " + selectedStrategy.toUpperCase();
        eyebrowText.textContent = text;
        statusDot.classList.toggle("spell-active", selectedSpell);
    }

    function updateModelButtons() {
        const locked = (selectedSpell && selectedLmMode !== "none") ? selectedLmMode : null;
        document.querySelectorAll("[data-model]").forEach(btn => {
            const avail = (AVAILABLE[btn.dataset.model] || []).length > 0;
            btn.disabled = !avail || (locked !== null && btn.dataset.model !== locked);
            btn.classList.toggle("active", btn.dataset.model === (locked || selectedModel));
        });
    }

    function activateToggle(group, attr, value) {
        group.querySelectorAll(".toggle-btn").forEach(btn => {
            btn.classList.toggle("active", btn.dataset[attr] === value);
        });
    }

    function updateDatasetAvailability() {
        const avail = AVAILABLE[selectedModel] || [];
        document.querySelectorAll("[data-dataset]").forEach(btn => {
            btn.disabled = !avail.includes(btn.dataset.dataset);
        });
        if (!avail.includes(selectedDataset)) selectedDataset = avail[0] || selectedDataset;
        activateToggle(document.getElementById("datasetGroup"), "dataset", selectedDataset);
    }

    function initAvailability() {
        updateModelButtons();
        updateDatasetAvailability();
        document.querySelectorAll("[data-lm]").forEach(btn => {
            const lm = btn.dataset.lm;
            if (lm === "ngram")       btn.disabled = !(AVAILABLE.ngram || []).length;
            if (lm === "transformer") btn.disabled = !(AVAILABLE.transformer || []).length;
        });
        updateDescriptions();
    }

    document.getElementById("modelGroup").addEventListener("click", e => {
        const btn = e.target.closest(".toggle-btn");
        if (!btn || btn.disabled) return;
        selectedModel = btn.dataset.model;
        updateModelButtons();
        updateDatasetAvailability();
        updateEyebrow();
        updateStrategyNote();
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

    document.getElementById("spellGroup").addEventListener("click", e => {
        const btn = e.target.closest(".toggle-btn");
        if (!btn) return;
        selectedSpell = btn.dataset.spell === "on";
        activateToggle(document.getElementById("spellGroup"), "spell", btn.dataset.spell);
        strategyRow.classList.toggle("visible", selectedSpell);
        lmRow.classList.toggle("visible", selectedSpell);
        weightRow.classList.toggle("visible", selectedSpell && selectedStrategy === "s3");
        modeText.classList.toggle("spell-mode", selectedSpell);
        updateModelButtons();
        updateEyebrow();
        updateStrategyNote();
        scheduleFetchSuggestions();
    });

    document.getElementById("strategyGroup").addEventListener("click", e => {
        const btn = e.target.closest(".toggle-btn");
        if (!btn || btn.disabled) return;
        selectedStrategy = btn.dataset.strategy;
        activateToggle(document.getElementById("strategyGroup"), "strategy", selectedStrategy);
        weightRow.classList.toggle("visible", selectedStrategy === "s3");
        updateDescriptions();
        updateEyebrow();
        updateStrategyNote();
        scheduleFetchSuggestions();
    });

    document.getElementById("lmGroup").addEventListener("click", e => {
        const btn = e.target.closest(".toggle-btn");
        if (!btn || btn.disabled) return;
        selectedLmMode = btn.dataset.lm;
        activateToggle(document.getElementById("lmGroup"), "lm", selectedLmMode);
        updateModelButtons();
        updateDescriptions();
        updateEyebrow();
        updateStrategyNote();
        scheduleFetchSuggestions();
    });

    document.getElementById("weightGroup").addEventListener("click", e => {
        const btn = e.target.closest(".toggle-btn");
        if (!btn) return;
        selectedWeight = btn.dataset.weight;
        activateToggle(document.getElementById("weightGroup"), "weight", selectedWeight);
        updateDescriptions();
        updateEyebrow();
        updateStrategyNote();
        scheduleFetchSuggestions();
    });

    function formatScore(s) {
        if (s < 0)     return s.toFixed(2);
        if (s < 0.001) return s.toExponential(2);
        return s.toFixed(4);
    }

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
        const base = parsed.prefix === "" ? "next-word prediction" : "prefix completion";
        modeText.textContent    = selectedSpell ? "spell · " + base : base;
        modeText.classList.toggle("spell-mode", selectedSpell);
        return parsed;
    }

    function clearSuggestions() { suggestionsDiv.innerHTML = ""; }

    function renderSuggestions(data) {
        suggestionsDiv.innerHTML = "";
        if (!data.suggestions || data.suggestions.length === 0) {
            suggestionsDiv.innerHTML = "<div class='empty'>No suggestions found. Try typing more context or a different prefix.</div>";
            return;
        }
        data.suggestions.forEach((item, index) => {
            const button = document.createElement("button");
            button.className = "suggestion-button";
            button.type = "button";

            const rankSpan = document.createElement("span");
            rankSpan.className = "rank";
            rankSpan.textContent = String(index + 1).padStart(2, "0");

            const wordSpan = document.createElement("span");
            wordSpan.className = "word";
            wordSpan.textContent = item.word;

            const scoreSpan = document.createElement("span");
            scoreSpan.className = "score";
            scoreSpan.textContent = formatScore(item.score);

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
        const params = getSpellParams();

        if (currentAbortController !== null) currentAbortController.abort();
        currentAbortController = new AbortController();
        statusText.textContent = "Updating...";

        try {
            const response = await fetch("/suggest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text:          text,
                    top_k:         7,
                    model:         params.model,
                    dataset:       selectedDataset,
                    spell:         selectedSpell,
                    weight_mode:   params.weight_mode,
                    max_edit_dist: params.max_edit_dist,
                    use_lm:        params.use_lm,
                }),
                signal: currentAbortController.signal,
            });

            if (!response.ok) throw new Error("Request failed.");
            const data = await response.json();
            if (requestId !== latestRequestId) return;

            contextText.textContent = JSON.stringify(data.context);
            prefixText.textContent  = data.prefix === "" ? "(empty)" : data.prefix;
            modeText.textContent    = data.mode;
            modeText.classList.toggle("spell-mode", data.spell && data.prefix !== "");

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

    initAvailability();
    updateEyebrow();
    updateStrategyNote();
    updateDescriptions();
    updateContextPrefixInstantly();
    fetchSuggestionsNow();
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML, available=AVAILABLE)


@app.route("/suggest", methods=["POST"])
def suggest():
    start_time = perf_counter()

    data          = request.get_json() or {}
    text          = data.get("text", "")
    top_k         = max(1, min(int(data.get("top_k", 5)), 10))
    model_key     = data.get("model", DEFAULT_MODEL)
    dataset_key   = data.get("dataset", DEFAULT_DATASET)
    spell_on      = bool(data.get("spell", False))
    weight_mode   = data.get("weight_mode", "uniform")
    max_edit_dist = int(data.get("max_edit_dist", 2))
    use_lm        = bool(data.get("use_lm", True))

    if model_key not in PREDICTORS or dataset_key not in PREDICTORS.get(model_key, {}):
        model_key   = DEFAULT_MODEL
        dataset_key = DEFAULT_DATASET

    context, prefix, suggestions, candidate_count = cached_suggest(
        text, top_k, model_key, dataset_key, spell_on, weight_mode, max_edit_dist, use_lm
    )
    elapsed_ms = (perf_counter() - start_time) * 1000

    if spell_on and prefix:
        mode = "spell · prefix completion"
    elif spell_on:
        mode = "spell · next-word prediction"
    elif prefix:
        mode = "prefix completion"
    else:
        mode = "next-word prediction"

    return jsonify({
        "context":         context,
        "prefix":          prefix,
        "mode":            mode,
        "candidate_count": candidate_count,
        "elapsed_ms":      elapsed_ms,
        "model":           model_key,
        "dataset":         dataset_key,
        "spell":           spell_on,
        "weight_mode":     weight_mode,
        "suggestions":     [{"word": w, "score": float(s)} for w, s in suggestions],
    })


if __name__ == "__main__":
    print(f"N-gram:      {AVAILABLE['ngram']}")
    print(f"Transformer: {AVAILABLE['transformer']}")
    app.run(debug=True, use_reloader=False, threaded=True)
