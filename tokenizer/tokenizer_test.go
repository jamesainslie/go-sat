package tokenizer

import (
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

	// Check vocab size (XLM-RoBERTa = 250000)
	if tok.VocabSize() != 250000 {
		t.Errorf("expected vocab size = 250000, got %d", tok.VocabSize())
	}

	// Check special token IDs based on actual model structure:
	// piece[0] = <unk> (UNKNOWN type)
	// piece[1] = <s> (CONTROL type - BOS)
	// piece[2] = </s> (CONTROL type - EOS)
	// Note: <pad> is not in the base SentencePiece model

	if tok.UnkID() != 0 {
		t.Errorf("expected UNK ID = 0, got %d", tok.UnkID())
	}
	if tok.BOSID() != 1 {
		t.Errorf("expected BOS ID = 1, got %d", tok.BOSID())
	}
	if tok.EOSID() != 2 {
		t.Errorf("expected EOS ID = 2, got %d", tok.EOSID())
	}
	// PadID should be -1 since <pad> is not in the base model
	if tok.PadID() != -1 {
		t.Errorf("expected PAD ID = -1 (not present), got %d", tok.PadID())
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
