#include "loomcore/tokenizer.h"

#include <cctype>
#include <fstream>
#include <sstream>

namespace loomcore {

namespace {
bool isAsciiPunct(unsigned char c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}
} // namespace

WordPieceTokenizer::WordPieceTokenizer(const std::string& vocab_path, size_t max_seq_len)
    : max_seq_len_(max_seq_len) {
    std::ifstream in(vocab_path);
    if (!in) throw LoomcoreError("WordPieceTokenizer: cannot open vocab file '" + vocab_path + "'");
    std::string line;
    int64_t id = 0;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        vocab_[line] = id;
        ++id;
    }
    if (vocab_.empty()) throw LoomcoreError("WordPieceTokenizer: vocab file '" + vocab_path + "' is empty");

    auto need = [&](const char* tok) -> int64_t {
        auto it = vocab_.find(tok);
        if (it == vocab_.end()) {
            throw LoomcoreError(std::string("WordPieceTokenizer: vocab is missing required special token '") + tok +
                                 "'");
        }
        return it->second;
    };
    cls_id_ = need("[CLS]");
    sep_id_ = need("[SEP]");
    pad_id_ = need("[PAD]");
    unk_id_ = need("[UNK]");
}

std::vector<std::string> WordPieceTokenizer::basicTokenize(const std::string& text) {
    // Lowercase (ASCII only — see header for scope) and split into
    // whitespace-delimited words, then split ASCII punctuation off into
    // its own tokens, mirroring BasicTokenizer's behavior for the plain
    // English text this pipeline actually produces (e.g. "a photo of a
    // golden retriever.").
    std::string lowered;
    lowered.reserve(text.size());
    for (unsigned char c : text) lowered.push_back(static_cast<char>(std::tolower(c)));

    std::vector<std::string> words;
    std::istringstream iss(lowered);
    std::string word;
    while (iss >> word) {
        std::string current;
        for (unsigned char c : word) {
            if (isAsciiPunct(c)) {
                if (!current.empty()) {
                    words.push_back(current);
                    current.clear();
                }
                words.emplace_back(1, static_cast<char>(c));
            } else {
                current.push_back(static_cast<char>(c));
            }
        }
        if (!current.empty()) words.push_back(current);
    }
    return words;
}

std::vector<std::string> WordPieceTokenizer::wordpiece(const std::string& token) const {
    // Greedy longest-match-first, the standard WordPiece algorithm: try to
    // consume the longest prefix that's in the vocab, prefixing "##" to
    // every piece after the first, then continue from where it left off.
    // Falls back to a single [UNK] if any piece can't be matched.
    std::vector<std::string> pieces;
    size_t start = 0;
    const size_t n = token.size();
    bool bad = false;
    while (start < n) {
        size_t end = n;
        std::string best;
        bool found = false;
        while (end > start) {
            std::string sub = (start == 0 ? token.substr(start, end - start) : "##" + token.substr(start, end - start));
            if (vocab_.count(sub)) {
                best = sub;
                found = true;
                break;
            }
            --end;
        }
        if (!found) {
            bad = true;
            break;
        }
        pieces.push_back(best);
        start = end;
    }
    if (bad || pieces.empty()) return {"[UNK]"};
    return pieces;
}

WordPieceTokenizer::Encoding WordPieceTokenizer::encode(const std::string& text) const {
    Encoding enc;
    std::vector<int64_t> ids;
    ids.push_back(cls_id_);
    for (const auto& word : basicTokenize(text)) {
        for (const auto& piece : wordpiece(word)) {
            auto it = vocab_.find(piece);
            ids.push_back(it == vocab_.end() ? unk_id_ : it->second);
            if (ids.size() >= max_seq_len_ - 1) break;
        }
        if (ids.size() >= max_seq_len_ - 1) break;
    }
    ids.push_back(sep_id_);

    enc.input_ids = ids;
    enc.token_type_ids.assign(ids.size(), 0);
    enc.attention_mask.assign(ids.size(), 1);
    while (enc.input_ids.size() < max_seq_len_) {
        enc.input_ids.push_back(pad_id_);
        enc.attention_mask.push_back(0);
        enc.token_type_ids.push_back(0);
    }
    enc.input_ids.resize(max_seq_len_);
    enc.attention_mask.resize(max_seq_len_);
    enc.token_type_ids.resize(max_seq_len_);
    return enc;
}

std::vector<NamedTensor> WordPieceTokenizer::encodeToTensors(const std::string& text) const {
    Encoding enc = encode(text);
    int64_t len = static_cast<int64_t>(enc.input_ids.size());
    std::vector<NamedTensor> tensors;
    tensors.push_back(NamedTensor::makeInt64("input_ids", {1, len}, enc.input_ids));
    tensors.push_back(NamedTensor::makeInt64("attention_mask", {1, len}, enc.attention_mask));
    tensors.push_back(NamedTensor::makeInt64("token_type_ids", {1, len}, enc.token_type_ids));
    return tensors;
}

} // namespace loomcore
