package tokenizer

import (
	"testing"

	pb "github.com/jamesainslie/go-sat/internal/proto"
)

func TestLoadModel(t *testing.T) {
	model, err := LoadModel("../testdata/sentencepiece.bpe.model")
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	// XLM-RoBERTa SentencePiece model has 250000 pieces
	// (the transformers library adds special tokens like <pad> separately)
	if len(model.Pieces) != 250000 {
		t.Errorf("expected 250000 pieces, got %d", len(model.Pieces))
	}

	// Check special tokens at their actual positions
	// piece[0] = <unk> (UNKNOWN type)
	// piece[1] = <s> (CONTROL type - BOS)
	// piece[2] = </s> (CONTROL type - EOS)
	if model.Pieces[0].Piece != "<unk>" {
		t.Errorf("expected piece[0] = <unk>, got %s", model.Pieces[0].Piece)
	}
	if model.Pieces[1].Piece != "<s>" {
		t.Errorf("expected piece[1] = <s>, got %s", model.Pieces[1].Piece)
	}
	if model.Pieces[2].Piece != "</s>" {
		t.Errorf("expected piece[2] = </s>, got %s", model.Pieces[2].Piece)
	}
}

func TestLoadModel_IsUnigram(t *testing.T) {
	model, err := LoadModel("../testdata/sentencepiece.bpe.model")
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	if model.TrainerSpec == nil {
		t.Fatal("expected trainer_spec to be present")
	}

	// Despite filename, XLM-RoBERTa uses UNIGRAM
	modelType := model.TrainerSpec.GetModelType()
	if modelType != pb.TrainerSpec_UNIGRAM {
		t.Errorf("expected UNIGRAM model type, got %v", modelType)
	}
}

func TestLoadModel_FileNotFound(t *testing.T) {
	_, err := LoadModel("../testdata/nonexistent.model")
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

func TestLoadModel_InvalidProtobuf(t *testing.T) {
	// Try to load a non-protobuf file
	_, err := LoadModel("../testdata/tokenizer_golden.json")
	if err == nil {
		t.Error("expected error for invalid protobuf data")
	}
}
