#!/usr/bin/env python3
"""Python-bindings demo (Milestone 4). Loads the same DAG as
examples/run_example.cpp (MobileNetV2 -> bert_tiny) through the pybind11
bindings, but implements the "skip bert_tiny once confident" routing rule
in *Python* (a loomcore.RoutingPolicy subclass, see SkipIfConfident below)
instead of C++ — showing the router is a genuine extension point from
either language, not just a C++ implementation detail wrapped for show.

Run from anywhere; it chdir's to the repo root itself. Requires the C++
extension to be built first — see bindings/python/loomcore/__init__.py's
docstring / README.md "Python bindings".
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bindings", "python"))
os.chdir(_REPO_ROOT)

import numpy as np  # noqa: E402

import loomcore  # noqa: E402


def load_labels():
    with open("assets/imagenet_labels.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class MiniWordPieceTokenizer:
    """A compact pure-Python mirror of loomcore::WordPieceTokenizer
    (src/tokenizer.cpp) — kept deliberately small. The point of this file
    is demonstrating the bindings and the Python-side router, not
    re-exporting the C++ tokenizer class; see docs/ARCHITECTURE.md
    "Python bindings" for why tokenization isn't bound directly."""

    def __init__(self, vocab_path: str, max_seq_len: int = 16):
        self.vocab = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i
        self.max_seq_len = max_seq_len

    def _wordpiece(self, token: str):
        pieces, start = [], 0
        while start < len(token):
            end = len(token)
            match = None
            while end > start:
                sub = token[start:end] if start == 0 else "##" + token[start:end]
                if sub in self.vocab:
                    match = sub
                    break
                end -= 1
            if match is None:
                return ["[UNK]"]
            pieces.append(match)
            start = end
        return pieces or ["[UNK]"]

    def encode(self, text: str):
        ids = [self.vocab["[CLS]"]]
        for word in text.lower().split():
            for piece in self._wordpiece(word):
                ids.append(self.vocab.get(piece, self.vocab["[UNK]"]))
                if len(ids) >= self.max_seq_len - 1:
                    break
            if len(ids) >= self.max_seq_len - 1:
                break
        ids.append(self.vocab["[SEP]"])
        mask = [1] * len(ids)
        pad = self.max_seq_len - len(ids)
        ids += [self.vocab["[PAD]"]] * pad
        mask += [0] * pad
        type_ids = [0] * self.max_seq_len
        return ids[: self.max_seq_len], mask[: self.max_seq_len], type_ids[: self.max_seq_len]


class SkipIfConfident(loomcore.RoutingPolicy):
    """Milestone 4's "basic router logic": skip the bert_tiny node once
    mobilenet is already confident. Functionally the same rule as C++'s
    ConfidenceGatePolicy (loomcore/router.h) but implemented here in
    Python and plugged into the same C++ scheduler via the RoutingPolicy
    trampoline — every job still calls back into this method from a
    scheduler worker thread (see loomcore_py.cpp's PyRoutingPolicy).

    Unlike the C++ policy, this one computes confidence itself straight
    from `ctx.upstream_confidence_source` (the raw mobilenet logits,
    exposed on RoutingContext precisely so a Python policy can do this)
    rather than delegating to a separate registered ConfidenceExtractor —
    a `confidence_extractors` entry is still registered in main() below
    because Runtime.load_graph requires one for any node that declares a
    "confidence_source" in its config, but its return value goes unused
    on this code path."""

    def __init__(self, threshold: float, labels: list, state: dict):
        super().__init__()
        self.threshold = threshold
        self.labels = labels
        self.state = state

    def decide(self, ctx: "loomcore.RoutingContext"):
        if ctx.node_id != "bert_tiny" or ctx.upstream_confidence_source is None:
            return None
        logits = ctx.upstream_confidence_source[0].reshape(-1)  # flatten [1, 1000] -> [1000]
        shifted = logits - np.max(logits)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        self.state["confidence"] = confidence
        self.state["label"] = self.labels[idx] if idx < len(self.labels) else "object"
        if confidence < self.threshold:
            return None
        return loomcore.RoutingDecision(skip=True, reason=f"python: confidence {confidence:.3f} >= {self.threshold}")


def main():
    labels = load_labels()
    tokenizer = MiniWordPieceTokenizer("models/bert_tiny_tokenizer/vocab.txt", max_seq_len=16)
    state = {"confidence": None, "label": None}

    def mobilenet_binder(graph_inputs, _upstream_outputs):
        return [("data", graph_inputs["data"])]

    def bert_binder(_graph_inputs, upstream_outputs):
        logits = upstream_outputs["mobilenet"][0]
        idx = int(np.argmax(logits))
        label = labels[idx] if idx < len(labels) else "object"
        text = f"a photo of a {label}"
        print(f'  [bert_tiny] embedding text: "{text}"')
        ids, mask, type_ids = tokenizer.encode(text)
        return [
            ("input_ids", np.array([ids], dtype=np.int64)),
            ("attention_mask", np.array([mask], dtype=np.int64)),
            ("token_type_ids", np.array([type_ids], dtype=np.int64)),
        ]

    def unused_confidence_extractor(_upstream_outputs):
        # Never actually called on this code path (see SkipIfConfident's
        # docstring) — registered only because load_graph requires *some*
        # ConfidenceExtractor for a node whose config sets
        # "confidence_source", as bert_tiny's does.
        return None

    router = loomcore.CompositeRouter()
    router.add(SkipIfConfident(0.85, labels, state))
    router.add(loomcore.LatencyBudgetPolicy(30.0))
    router.add(loomcore.LoadAwareBackendPolicy())

    runtime = loomcore.Runtime()
    runtime.load_graph(
        "examples/graph_config.json",
        binders={"mobilenet": mobilenet_binder, "bert_tiny": bert_binder},
        confidence_extractors={"bert_tiny": unused_confidence_extractor},
        router=router,
    )

    # A deterministic synthetic image (no PIL/stb dependency needed on the
    # Python side) — see examples/run_example.cpp for the real-photo path.
    rng = np.random.RandomState(0)
    image = rng.uniform(-1.0, 1.0, size=(1, 3, 224, 224)).astype(np.float32)

    print("\n=== Single run via Python bindings ===")
    result = runtime.run({"data": image})
    print(f"Predicted class : {state['label']} (confidence {state['confidence']:.3f})")
    if "bert_tiny" in result and result["bert_tiny"]:
        embedding = result["bert_tiny"][1]  # [last_hidden_state, pooler_output]
        print(f"Text embedding  : shape={embedding.shape} first values={embedding[0, :4]}")
    else:
        print("bert_tiny was skipped by the Python SkipIfConfident policy.")

    stats = runtime.node_stats("mobilenet")
    print(f"\nmobilenet latency: p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms n={stats['count']}")

    print("\nLast 3 structured log lines:")
    for line in runtime.recent_logs(3):
        print(" ", line)


if __name__ == "__main__":
    main()
