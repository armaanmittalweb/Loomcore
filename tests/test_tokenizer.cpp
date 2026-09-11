#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include "loomcore/tokenizer.h"

using namespace loomcore;

namespace {
// A hand-built miniature vocab so this test never depends on the real
// (network-fetched) BERT-tiny vocab.txt — see docs/ARCHITECTURE.md
// "Tokenizer scope".
std::string writeTestVocab() {
    auto path = std::filesystem::temp_directory_path() / "loomcore_test_vocab.txt";
    std::ofstream out(path, std::ios::trunc);
    for (const char* tok : {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "photo", "of", "golden", "retriever", "dog", "##s"}) {
        out << tok << "\n";
    }
    out.close();
    return path.string();
}
} // namespace

TEST_CASE("WordPieceTokenizer encodes whole-word matches with [CLS]/[SEP] and padding") {
    WordPieceTokenizer tok(writeTestVocab(), /*max_seq_len=*/16);
    auto enc = tok.encode("a photo of a golden retriever");

    REQUIRE(enc.input_ids.size() == 16);
    // [CLS] a photo of a golden retriever [SEP] PAD...
    std::vector<int64_t> expected_prefix = {2, 4, 5, 6, 4, 7, 8, 3};
    for (size_t i = 0; i < expected_prefix.size(); ++i) CHECK(enc.input_ids[i] == expected_prefix[i]);
    for (size_t i = expected_prefix.size(); i < enc.input_ids.size(); ++i) CHECK(enc.input_ids[i] == 0 /* [PAD] */);

    for (size_t i = 0; i < expected_prefix.size(); ++i) CHECK(enc.attention_mask[i] == 1);
    for (size_t i = expected_prefix.size(); i < enc.attention_mask.size(); ++i) CHECK(enc.attention_mask[i] == 0);

    for (auto tt : enc.token_type_ids) CHECK(tt == 0);
}

TEST_CASE("WordPieceTokenizer splits an unseen word into known subwords") {
    WordPieceTokenizer tok(writeTestVocab(), 8);
    auto enc = tok.encode("dogs");
    // [CLS] dog ##s [SEP] PAD PAD PAD PAD
    CHECK(enc.input_ids[0] == 2); // [CLS]
    CHECK(enc.input_ids[1] == 9); // dog
    CHECK(enc.input_ids[2] == 10); // ##s
    CHECK(enc.input_ids[3] == 3); // [SEP]
}

TEST_CASE("WordPieceTokenizer falls back to [UNK] for a completely unknown word") {
    WordPieceTokenizer tok(writeTestVocab(), 8);
    auto enc = tok.encode("xyzzy");
    CHECK(enc.input_ids[0] == 2); // [CLS]
    CHECK(enc.input_ids[1] == 1); // [UNK]
    CHECK(enc.input_ids[2] == 3); // [SEP]
}

TEST_CASE("WordPieceTokenizer truncates to max_seq_len, always keeping [SEP] last-in-content") {
    WordPieceTokenizer tok(writeTestVocab(), 6);
    auto enc = tok.encode("a photo of a golden retriever");
    REQUIRE(enc.input_ids.size() == 6);
    std::vector<int64_t> expected = {2, 4, 5, 6, 4, 3}; // truncated before "golden retriever"
    CHECK(enc.input_ids == expected);
}

TEST_CASE("WordPieceTokenizer::encodeToTensors packs shape [1, max_seq_len]") {
    WordPieceTokenizer tok(writeTestVocab(), 8);
    auto tensors = tok.encodeToTensors("a photo");
    REQUIRE(tensors.size() == 3);
    for (const auto& t : tensors) {
        CHECK(t.shape == std::vector<int64_t>{1, 8});
        // Cast to int: see the comment above LatencyBudgetPolicy's test in
        // test_router.cpp for why enum-class CHECK()s go through int here.
        CHECK(static_cast<int>(t.dtype) == static_cast<int>(DType::Int64));
    }
    CHECK(tensors[0].name == "input_ids");
    CHECK(tensors[1].name == "attention_mask");
    CHECK(tensors[2].name == "token_type_ids");
}
