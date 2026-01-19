package tokenizer

import (
	"fmt"

	pb "github.com/jamesainslie/go-sat/internal/proto"
)

// Tokenizer implements XLM-RoBERTa compatible SentencePiece Unigram tokenization.
type Tokenizer struct {
	pieces      map[string]int32   // token string -> ID
	scores      map[string]float32 // token string -> log probability
	idToPiece   []string           // ID -> token string
	pieceToType map[string]pb.ModelProto_SentencePiece_Type

	bosID int32
	padID int32
	eosID int32
	unkID int32

	maxTokenLen int
}

// TokenInfo represents a token with its position in the original text.
type TokenInfo struct {
	ID    int32
	Text  string
	Start int // byte offset in original text
	End   int // byte offset in original text
}

// New loads a tokenizer from a SentencePiece .model file.
func New(modelPath string) (*Tokenizer, error) {
	model, err := LoadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading model: %w", err)
	}

	t := &Tokenizer{
		pieces:      make(map[string]int32),
		scores:      make(map[string]float32),
		idToPiece:   make([]string, len(model.Pieces)),
		pieceToType: make(map[string]pb.ModelProto_SentencePiece_Type),
		bosID:       -1,
		padID:       -1,
		eosID:       -1,
		unkID:       -1,
	}

	for i, piece := range model.Pieces {
		id := int32(i)
		pieceStr := piece.Piece

		t.pieces[pieceStr] = id
		t.scores[pieceStr] = piece.Score
		t.idToPiece[i] = pieceStr
		t.pieceToType[pieceStr] = piece.Type

		// Track max token length for optimization
		if len(pieceStr) > t.maxTokenLen {
			t.maxTokenLen = len(pieceStr)
		}

		// Identify special tokens by checking their type and string
		switch piece.Type {
		case pb.ModelProto_SentencePiece_CONTROL:
			switch pieceStr {
			case "<s>":
				t.bosID = id
			case "</s>":
				t.eosID = id
			case "<pad>":
				t.padID = id
			}
		case pb.ModelProto_SentencePiece_UNKNOWN:
			t.unkID = id
		}
	}

	return t, nil
}

// Close releases tokenizer resources.
func (t *Tokenizer) Close() error {
	return nil
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int {
	return len(t.idToPiece)
}

// BOSID returns the beginning-of-sentence token ID.
func (t *Tokenizer) BOSID() int32 { return t.bosID }

// PadID returns the padding token ID.
func (t *Tokenizer) PadID() int32 { return t.padID }

// EOSID returns the end-of-sentence token ID.
func (t *Tokenizer) EOSID() int32 { return t.eosID }

// UnkID returns the unknown token ID.
func (t *Tokenizer) UnkID() int32 { return t.unkID }
