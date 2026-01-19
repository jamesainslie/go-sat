package tokenizer

import (
	"encoding/json"
	"os"
	"testing"
)

func TestNew(t *testing.T) {
	tok, err := New("../testdata/sentencepiece.bpe.model")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer func() {
		if err := tok.Close(); err != nil {
			t.Errorf("Close failed: %v", err)
		}
	}()

	// Check vocab size (HuggingFace XLM-RoBERTa = 250002)
	// SentencePiece has 250000, HuggingFace adds 2 for ID remapping
	if tok.VocabSize() != 250002 {
		t.Errorf("expected vocab size = 250002, got %d", tok.VocabSize())
	}

	// Check special token IDs (HuggingFace XLM-RoBERTa convention):
	// HF[0] = <s>   (BOS)
	// HF[1] = <pad> (PAD)
	// HF[2] = </s>  (EOS)
	// HF[3] = <unk> (UNK)

	if tok.BOSID() != 0 {
		t.Errorf("expected BOS ID = 0, got %d", tok.BOSID())
	}
	if tok.PadID() != 1 {
		t.Errorf("expected PAD ID = 1, got %d", tok.PadID())
	}
	if tok.EOSID() != 2 {
		t.Errorf("expected EOS ID = 2, got %d", tok.EOSID())
	}
	if tok.UnkID() != 3 {
		t.Errorf("expected UNK ID = 3, got %d", tok.UnkID())
	}
}

func TestNew_FileNotFound(t *testing.T) {
	_, err := New("../testdata/nonexistent.model")
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

func TestTokenizer_EncodeIDs_Simple(t *testing.T) {
	tok, err := New("../testdata/sentencepiece.bpe.model")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer func() {
		if err := tok.Close(); err != nil {
			t.Errorf("Close failed: %v", err)
		}
	}()

	// Simple case - just verify we get some tokens
	ids := tok.EncodeIDs("Hello")
	if len(ids) == 0 {
		t.Error("expected non-empty token IDs")
	}

	// All IDs should be valid
	for i, id := range ids {
		if id < 0 || int(id) >= tok.VocabSize() {
			t.Errorf("token %d: invalid ID %d", i, id)
		}
	}
}

type goldenCase struct {
	Input    string   `json:"input"`
	TokenIDs []int    `json:"token_ids"`
	Offsets  [][2]int `json:"offsets"`
	Tokens   []string `json:"tokens"`
}

func loadGoldenCases(t *testing.T) []goldenCase {
	t.Helper()
	data, err := os.ReadFile("../testdata/tokenizer_golden.json")
	if err != nil {
		t.Fatalf("loading golden file: %v", err)
	}
	var cases []goldenCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("parsing golden file: %v", err)
	}
	return cases
}

func TestTokenizer_EncodeIDs_Golden(t *testing.T) {
	tok, err := New("../testdata/sentencepiece.bpe.model")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer func() {
		if err := tok.Close(); err != nil {
			t.Errorf("Close failed: %v", err)
		}
	}()

	cases := loadGoldenCases(t)
	for _, tc := range cases {
		t.Run(tc.Input, func(t *testing.T) {
			if tc.Input == "" {
				// Skip empty string - handled separately
				return
			}

			got := tok.EncodeIDs(tc.Input)

			if len(got) != len(tc.TokenIDs) {
				t.Errorf("length mismatch: got %d tokens, want %d", len(got), len(tc.TokenIDs))
				t.Logf("got:  %v", got)
				t.Logf("want: %v", tc.TokenIDs)
				return
			}

			for i := range got {
				if int(got[i]) != tc.TokenIDs[i] {
					t.Errorf("token %d: got ID %d, want %d", i, got[i], tc.TokenIDs[i])
				}
			}
		})
	}
}

func TestTokenizer_Decode(t *testing.T) {
	tok, err := New("../testdata/sentencepiece.bpe.model")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer func() {
		if err := tok.Close(); err != nil {
			t.Errorf("Close failed: %v", err)
		}
	}()

	tests := []struct {
		input    string
		expected string
	}{
		{"Hello", "Hello"},
		{"Hello world", "Hello world"},
	}

	for _, tc := range tests {
		ids := tok.EncodeIDs(tc.input)
		got := tok.Decode(ids)
		if got != tc.expected {
			t.Errorf("Decode(Encode(%q)) = %q, want %q", tc.input, got, tc.expected)
		}
	}
}
