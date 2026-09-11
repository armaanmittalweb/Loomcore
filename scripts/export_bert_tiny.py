#!/usr/bin/env python3
"""Exports prajjwal1/bert-tiny (4.4M params — the "tiny DistilBERT-variant"
slot in the project spec, chosen because it's smaller and faster than
distilbert-base while exercising the exact same BERT-family ONNX I/O
shape) to ONNX with a dynamic batch AND sequence-length axis, using a
plain manual torch.onnx.export rather than a higher-level exporter
package.

No training happens here — this is the "don't train anything, the
orchestration layer is the point" model straight off the shelf.

Note: this repo predates transformers' `model_type` config convention and
ships only a legacy (non-`tokenizers`-library) vocab.txt, so `AutoModel` /
`AutoTokenizer` both refuse it on transformers 5.x. We load the model
directly via `BertModel` (its config.json is otherwise a completely
standard BERT config) and fetch vocab.txt straight from the Hub — it's
all Loomcore's own C++ WordPieceTokenizer (tokenizer.h/.cpp) needs.
"""
import os

import torch
from huggingface_hub import hf_hub_download
from transformers import BertConfig, BertModel

MODEL_NAME = "prajjwal1/bert-tiny"
ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")


def main() -> None:
    config = BertConfig.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME, config=config)
    model.eval()
    model.config.return_dict = False  # torch.onnx.export needs a plain tuple output

    seq_len = 12
    dummy_ids = torch.randint(low=0, high=config.vocab_size, size=(1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((1, seq_len), dtype=torch.long)
    dummy_type_ids = torch.zeros((1, seq_len), dtype=torch.long)

    os.makedirs(MODELS_DIR, exist_ok=True)
    onnx_path = os.path.join(MODELS_DIR, "bert_tiny.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask, dummy_type_ids),
            onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
                "pooler_output": {0: "batch"},
            },
            opset_version=17,
        )
    print(f"exported {onnx_path}")

    vocab_dir = os.path.join(MODELS_DIR, "bert_tiny_tokenizer")
    os.makedirs(vocab_dir, exist_ok=True)
    vocab_src = hf_hub_download(repo_id=MODEL_NAME, filename="vocab.txt")
    with open(vocab_src, "rb") as src, open(os.path.join(vocab_dir, "vocab.txt"), "wb") as dst:
        dst.write(src.read())
    print(f"saved vocab.txt to {vocab_dir}")


if __name__ == "__main__":
    main()
