// Loomcore — a minimal WordPiece tokenizer for the BERT-tiny DAG node.
//
// This is a small, self-contained, from-scratch implementation (no ICU /
// no Hugging Face `tokenizers` dependency) so the C++ core has no
// third-party NLP dependency. It covers the common case the reference
// pipeline actually needs — lowercase ASCII-ish text such as
// "a photo of a golden retriever" — via whitespace + ASCII punctuation
// splitting followed by greedy longest-match WordPiece. It does NOT
// implement full Unicode normalization, CJK character splitting, or
// accent stripping the way the reference Python tokenizer does; anything
// outside its scope falls back to [UNK]. See docs/ARCHITECTURE.md
// "Tokenizer scope" for the exact limitations.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "loomcore/export.h"
#include "loomcore/types.h"

namespace loomcore {

class LOOMCORE_API WordPieceTokenizer {
public:
    // `vocab_path` is a newline-delimited vocab.txt (one token per line,
    // line number == token id), the format `transformers`'
    // `PreTrainedTokenizer.save_pretrained` writes.
    explicit WordPieceTokenizer(const std::string& vocab_path, size_t max_seq_len = 32);

    struct Encoding {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> token_type_ids;
    };

    // Encodes a single sequence (no sentence-pair support — the reference
    // pipeline only ever embeds one short label phrase at a time): adds
    // [CLS]/[SEP], pads or truncates to max_seq_len.
    Encoding encode(const std::string& text) const;

    // Packs an Encoding into three NamedTensors of shape [1, max_seq_len],
    // named "input_ids", "attention_mask", "token_type_ids" — the
    // conventional Hugging Face BERT export input names.
    std::vector<NamedTensor> encodeToTensors(const std::string& text) const;

    size_t vocabSize() const { return vocab_.size(); }

private:
    std::unordered_map<std::string, int64_t> vocab_;
    size_t max_seq_len_;
    int64_t cls_id_ = 0, sep_id_ = 0, pad_id_ = 0, unk_id_ = 0;

    static std::vector<std::string> basicTokenize(const std::string& text);
    std::vector<std::string> wordpiece(const std::string& token) const;
};

} // namespace loomcore
