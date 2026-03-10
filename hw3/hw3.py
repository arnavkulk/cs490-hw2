#!/usr/bin/env python3
"""
hw3.py — Syntactic Complexity and Model Robustness
CS 490: Natural Language Processing · Spring 2026

# requirements.txt
# ─────────────────
# datasets>=2.14
# transformers>=4.30
# torch>=2.0
# spacy>=3.5
# pandas>=2.0
# numpy>=1.24
#
# Also run:  python -m spacy download en_core_web_sm
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import spacy
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

MNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}

DEFAULT_MODELS = [
    "typeform/distilbert-base-uncased-mnli",
    "typeform/mobilebert-uncased-mnli",
    "Alireza1044/albert-base-v2-mnli",
    "vish88/xlnet-base-mnli-finetuned",
]

PERTURBATION_METHODS = [
    "adjective_to_relative_clause",
    "parenthetical_insertion",
    "appositive_person_insertion",
]

COMPLEXITY_METRICS = [
    "token_length",
    "max_dependency_depth",
    "mean_dependency_arc_length",
]

PARENTHETICAL_PHRASES = [
    ", as was widely known,",
    ", according to the report,",
    ", which was mentioned earlier,",
]

ANIMATE_NOUNS = frozenset({
    "man", "woman", "boy", "girl", "person", "child", "people",
    "student", "teacher", "doctor", "player", "author", "artist",
    "officer", "soldier", "worker", "king", "queen", "prince",
    "princess", "friend", "enemy", "leader", "minister", "president",
    "husband", "wife", "father", "mother", "son", "daughter",
})

# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_jsonl(records: list[dict], path: str):
    """Write a list of dicts as a JSONL file."""
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    print(f"  Saved {len(records)} records -> {path}")


def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def _stable_hash(text: str) -> int:
    """Deterministic hash independent of PYTHONHASHSEED."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset sampling
# ═══════════════════════════════════════════════════════════════════════════


def sample_dataset(n_per_split: int, seed: int) -> list[dict]:
    """Sample equally from MultiNLI matched and mismatched dev sets."""
    print("Loading MultiNLI dataset...")
    ds = load_dataset("multi_nli")

    examples = []
    for split_tag, hf_key in [
        ("matched", "validation_matched"),
        ("mismatched", "validation_mismatched"),
    ]:
        pool = ds[hf_key]
        indices = list(range(len(pool)))
        random.Random(seed).shuffle(indices)
        for idx in indices[:n_per_split]:
            row = pool[idx]
            examples.append(
                {
                    "id": f"{split_tag}_{idx}",
                    "split": split_tag,
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": int(row["label"]),
                }
            )

    print(f"  Sampled {len(examples)} examples ({n_per_split} per split)")
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# Complexity metrics
# ═══════════════════════════════════════════════════════════════════════════


def _dep_depth(token) -> int:
    """Walk up the dependency tree to root, counting edges."""
    d = 0
    cur = token
    while cur.head != cur:
        d += 1
        cur = cur.head
    return d


def compute_metrics(doc) -> dict[str, float]:
    """Compute all complexity metrics for a single spaCy Doc."""
    n = len(doc)
    if n == 0:
        return dict.fromkeys(COMPLEXITY_METRICS, 0.0)

    depths = [_dep_depth(t) for t in doc]
    arcs = [abs(t.i - t.head.i) for t in doc if t.head != t]

    return {
        "token_length": float(n),
        "max_dependency_depth": float(max(depths)),
        "mean_dependency_arc_length": float(np.mean(arcs)) if arcs else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Perturbations
# ═══════════════════════════════════════════════════════════════════════════


def _rebuild(
    doc, skip: set[int] | None = None, inserts: dict[int, str] | None = None
) -> str:
    """Reconstruct sentence from spaCy Doc, optionally skipping tokens and
    inserting text after specified token indices."""
    skip = skip or set()
    inserts = inserts or {}
    parts: list[str] = []
    for tok in doc:
        if tok.i in skip:
            continue
        parts.append(tok.text)
        if tok.i in inserts:
            parts.append(inserts[tok.i])
        parts.append(tok.whitespace_)
    return re.sub(r" {2,}", " ", "".join(parts)).strip()


def perturb_adjective_to_relative_clause(doc) -> str | None:
    """Convert an adjectival modifier into a relative clause.

    'The tall man walked' -> 'The man, who is tall, walked'
    """
    for tok in doc:
        if tok.dep_ != "amod" or tok.pos_ != "ADJ":
            continue
        noun = tok.head
        if noun.pos_ not in ("NOUN", "PROPN") or tok.i >= noun.i:
            continue

        sub_tokens = sorted(tok.subtree, key=lambda t: t.i)
        sub_ids = {t.i for t in sub_tokens if t.i < noun.i}
        if not sub_ids:
            continue

        phrase = " ".join(t.text for t in sub_tokens if t.i in sub_ids)

        animate = noun.text.lower() in ANIMATE_NOUNS or any(
            e.label_ == "PERSON" for e in doc.ents if e.start <= noun.i < e.end
        )
        rel = "who" if animate else "which"
        clause = f", {rel} is {phrase},"

        result = _rebuild(doc, skip=sub_ids, inserts={noun.i: clause})
        if result != doc.text.strip():
            return result

    return None


def perturb_parenthetical_insertion(doc) -> str | None:
    """Insert a semantically-irrelevant parenthetical after the subject NP
    or, as a fallback, after the first named entity."""
    anchor = None
    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass"):
            anchor = max(t.i for t in tok.subtree)
            break

    if anchor is None and doc.ents:
        anchor = doc.ents[0].end - 1

    if anchor is None or anchor >= len(doc) - 1:
        return None

    phrase = PARENTHETICAL_PHRASES[
        _stable_hash(doc.text) % len(PARENTHETICAL_PHRASES)
    ]
    result = _rebuild(doc, inserts={anchor: phrase})
    return result if result != doc.text.strip() else None


def perturb_appositive_person(doc) -> str | None:
    """Insert an appositive after the first PERSON entity.

    'John arrived' -> 'John, a person, arrived'
    """
    persons = [e for e in doc.ents if e.label_ == "PERSON"]
    if not persons:
        return None

    anchor = persons[0].end - 1
    if anchor >= len(doc) - 1:
        return None

    result = _rebuild(doc, inserts={anchor: ", a person,"})
    return result if result != doc.text.strip() else None


PERTURBATION_FNS = {
    "adjective_to_relative_clause": perturb_adjective_to_relative_clause,
    "parenthetical_insertion": perturb_parenthetical_insertion,
    "appositive_person_insertion": perturb_appositive_person,
}


def apply_perturbations(
    examples: list[dict], nlp, output_dir: str
) -> dict[str, list[dict]]:
    """Apply every perturbation type to premises.

    Returns a dict mapping perturbation name -> list of perturbed examples.
    Caches results to JSONL files for resume support.
    """
    print("  Parsing premises...")
    docs = list(nlp.pipe([ex["premise"] for ex in examples], batch_size=512))

    results: dict[str, list[dict]] = {}
    for method in PERTURBATION_METHODS:
        cache_path = os.path.join(output_dir, f"perturbed_{method}.jsonl")
        if os.path.exists(cache_path):
            print(f"  [{method}] loaded from cache")
            results[method] = load_jsonl(cache_path)
            continue

        fn = PERTURBATION_FNS[method]
        print(f"  [{method}] applying...")
        perturbed: list[dict] = []
        for ex, doc in zip(examples, docs):
            new_premise = fn(doc)
            if new_premise is not None:
                perturbed.append(
                    {
                        **ex,
                        "original_premise": ex["premise"],
                        "premise": new_premise,
                        "perturbation": method,
                    }
                )

        pct = 100 * len(perturbed) / len(examples) if examples else 0
        print(f"    {len(perturbed)}/{len(examples)} succeeded ({pct:.1f}%)")
        results[method] = perturbed
        save_jsonl(perturbed, cache_path)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Model inference
# ═══════════════════════════════════════════════════════════════════════════


def _normalize_label(s: str) -> int | None:
    """Map a model label string to 0/1/2 (entailment/neutral/contradiction)."""
    s = s.lower().strip()
    direct = {"entailment": 0, "neutral": 1, "contradiction": 2}
    if s in direct:
        return direct[s]
    if s.startswith("label_"):
        try:
            return int(s.split("_")[1])
        except (IndexError, ValueError):
            pass
    return None


def _build_label_map(model) -> dict[int, int]:
    """Build model-output-index -> standard NLI label id mapping."""
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        return {0: 0, 1: 1, 2: 2}
    lm: dict[int, int] = {}
    for k, v in id2label.items():
        nli = _normalize_label(str(v))
        if nli is not None:
            lm[int(k)] = nli
    return lm if len(lm) == 3 else {0: 0, 1: 1, 2: 2}


def _infer_batched(model, tokenizer, label_map, examples, batch_size, device):
    """Run batched inference, returning predicted NLI label ids."""
    preds: list[int] = []
    total = (len(examples) + batch_size - 1) // batch_size
    for bi in range(0, len(examples), batch_size):
        batch = examples[bi : bi + batch_size]
        enc = tokenizer(
            [e["premise"] for e in batch],
            [e["hypothesis"] for e in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        raw = logits.argmax(-1).cpu().tolist()
        preds.extend(label_map.get(p, p) for p in raw)

        step = bi // batch_size + 1
        if step % 25 == 0 or step == total:
            print(f"      batch {step}/{total}")

    return preds


def evaluate_models(
    model_names: list[str],
    original: list[dict],
    perturbed: dict[str, list[dict]],
    batch_size: int,
    device: torch.device,
    output_dir: str,
) -> list[dict]:
    """Evaluate every model on original + every perturbed subset.

    Uses file-based prediction caching so partial runs can be resumed.
    Loads each model only once across all subsets.
    """
    perf_rows: list[dict] = []
    subsets = [("original", original)] + [
        (m, exs) for m, exs in perturbed.items() if exs
    ]

    for model_name in model_names:
        short = model_name.rsplit("/", 1)[-1]
        safe = model_name.replace("/", "__")
        print(f"\n  Model: {short}")

        tags = [tag for tag, _ in subsets]
        all_cached = all(
            os.path.exists(os.path.join(output_dir, f"preds_{safe}_{t}.json"))
            for t in tags
        )

        model = tokenizer = label_map = None
        if not all_cached:
            try:
                print(f"    Loading {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                model.to(device).eval()
                label_map = _build_label_map(model)
            except Exception as exc:
                print(f"    Failed to load: {exc}")
                continue

        for tag, exs in subsets:
            cache = os.path.join(output_dir, f"preds_{safe}_{tag}.json")

            if os.path.exists(cache):
                with open(cache) as f:
                    preds = json.load(f)
                print(f"    [{tag}] loaded predictions from cache")
            else:
                print(
                    f"    [{tag}] running inference ({len(exs)} examples)..."
                )
                preds = _infer_batched(
                    model, tokenizer, label_map, exs, batch_size, device
                )
                with open(cache, "w") as f:
                    json.dump(preds, f)

            labels = [e["label"] for e in exs]
            n_correct = sum(
                p == lab for p, lab in zip(preds, labels)
            )
            acc = n_correct / len(labels) if labels else 0.0
            print(f"    [{tag}] accuracy = {acc:.4f}")
            perf_rows.append(
                {
                    "model": model_name,
                    "perturbation_method": tag,
                    "performance": round(acc, 4),
                }
            )

        if model is not None:
            del model, tokenizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return perf_rows


# ═══════════════════════════════════════════════════════════════════════════
# Complexity aggregation
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_complexity(
    original: list[dict],
    perturbed: dict[str, list[dict]],
    nlp,
) -> list[dict]:
    """Compute mean complexity metrics for original and each perturbation."""
    rows: list[dict] = []
    all_subsets = [("original", original)] + list(perturbed.items())

    for tag, exs in all_subsets:
        if not exs:
            continue
        print(f"  [{tag}] computing metrics ({len(exs)} examples)...")
        docs = list(nlp.pipe([e["premise"] for e in exs], batch_size=512))
        all_m = [compute_metrics(d) for d in docs]
        for metric in COMPLEXITY_METRICS:
            vals = [m[metric] for m in all_m]
            rows.append(
                {
                    "perturbation_method": tag,
                    "metric_type": metric,
                    "value": round(float(np.mean(vals)), 4),
                }
            )

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="HW3: Syntactic Complexity & Model Robustness",
    )
    parser.add_argument("--sample_size_per_split", type=int, default=7500)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap total examples (useful for quick tests)",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug", action="store_true", help="Fast run with ~100 examples"
    )
    args = parser.parse_args()

    if args.debug:
        args.sample_size_per_split = min(args.sample_size_per_split, 50)
        args.max_examples = args.max_examples or 100

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ── spaCy ──────────────────────────────────────────────────────────────
    print("\nLoading spaCy...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("  Downloading en_core_web_sm...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # ── Dataset ────────────────────────────────────────────────────────────
    print("\n== Dataset ==")
    data_path = os.path.join(args.output_dir, "data.jsonl")
    if os.path.exists(data_path):
        print(f"Loading cached dataset from {data_path}")
        examples = load_jsonl(data_path)
    else:
        examples = sample_dataset(args.sample_size_per_split, args.seed)
        save_jsonl(examples, data_path)

    if args.max_examples:
        examples = examples[: args.max_examples]
        print(f"  (capped to {len(examples)} examples)")

    # ── Perturbations ──────────────────────────────────────────────────────
    print("\n== Perturbations ==")
    perturbed = apply_perturbations(examples, nlp, args.output_dir)

    # ── Model evaluation ───────────────────────────────────────────────────
    print("\n== Model Evaluation ==")
    perf_rows = evaluate_models(
        args.models,
        examples,
        perturbed,
        args.batch_size,
        device,
        args.output_dir,
    )

    # ── Complexity ─────────────────────────────────────────────────────────
    print("\n== Complexity Metrics ==")
    complex_rows = aggregate_complexity(examples, perturbed, nlp)

    # ── Save CSVs ──────────────────────────────────────────────────────────
    perf_df = pd.DataFrame(perf_rows)
    complex_df = pd.DataFrame(complex_rows)

    perf_csv = os.path.join(args.output_dir, "perf.csv")
    complex_csv = os.path.join(args.output_dir, "complex.csv")
    perf_df.to_csv(perf_csv, index=False)
    complex_df.to_csv(complex_csv, index=False)

    # Copy data.jsonl to repo root for submission
    root_data = "data.jsonl"
    if os.path.abspath(data_path) != os.path.abspath(root_data):
        shutil.copy(data_path, root_data)

    print(f"\n{'=' * 60}")
    print("Saved:")
    print(f"  {perf_csv}")
    print(f"  {complex_csv}")
    print(f"  {data_path}  (also copied to ./data.jsonl)")
    print(f"{'=' * 60}")

    print("\n-- Performance --")
    print(perf_df.to_string(index=False))
    print("\n-- Complexity --")
    print(complex_df.to_string(index=False))


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# HOW TO RUN
# ═══════════════════════════════════════════════════════════════════════════
#
# 1. Install dependencies:
#        pip install datasets transformers torch spacy pandas numpy
#        python -m spacy download en_core_web_sm
#
# 2. Full run (15,000 examples, 4 models):
#        python hw3.py
#
# 3. Debug / quick smoke test (~100 examples):
#        python hw3.py --debug
#
# 4. Custom configuration:
#        python hw3.py --sample_size_per_split 5000 --batch_size 16 --seed 7
#
# EXPECTED OUTPUTS (in ./output/ by default):
#   perf.csv              – accuracy per model x perturbation
#   complex.csv           – mean complexity per perturbation x metric
#   data.jsonl            – sampled dataset
#   perturbed_*.jsonl     – cached perturbed datasets
#   preds_*.json          – cached per-model predictions
#
# data.jsonl is also copied to the repo root for Gradescope submission.
# The script is resume-friendly: re-run it and cached files are reused.
