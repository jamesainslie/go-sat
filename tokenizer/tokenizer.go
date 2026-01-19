package tokenizer

import (
	"fmt"

	pb "github.com/jamesainslie/go-sat/internal/proto"
)

// Tokenizer implements XLM-RoBERTa compatible SentencePiece Unigram tokenization.
//
// Note: Token IDs are remapped from SentencePiece indices to match HuggingFace
// XLM-RoBERTa convention:
//   - HF[0] = <s>   (SP[1])
//   - HF[1] = <pad> (not in SentencePiece)
//   - HF[2] = </s>  (SP[2])
//   - HF[3] = <unk> (SP[0])
//   - HF[n+1] = SP[n] for n >= 3 (normal tokens shifted by 1)
type Tokenizer struct {
	pieces      map[string]int32   // token string -> SentencePiece index (internal use)
	scores      map[string]float32 // token string -> log probability
	idToPiece   []string           // SentencePiece index -> token string
	pieceToType map[string]pb.ModelProto_SentencePiece_Type

	// HuggingFace-compatible token IDs
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
		// HuggingFace XLM-RoBERTa special token IDs
		bosID: 0, // <s>
		padID: 1, // <pad>
		eosID: 2, // </s>
		unkID: 3, // <unk>
	}

	for i, piece := range model.Pieces {
		pieceStr := piece.Piece

		// Store SentencePiece index for internal Viterbi algorithm
		t.pieces[pieceStr] = int32(i)
		t.scores[pieceStr] = piece.Score
		t.idToPiece[i] = pieceStr
		t.pieceToType[pieceStr] = piece.Type

		// Track max token length for optimization
		if len(pieceStr) > t.maxTokenLen {
			t.maxTokenLen = len(pieceStr)
		}
	}

	return t, nil
}

// spIndexToHFID converts a SentencePiece index to a HuggingFace XLM-RoBERTa token ID.
//
// Mapping:
//   - SP[0] (<unk>) -> HF[3]
//   - SP[1] (<s>)   -> HF[0]
//   - SP[2] (</s>)  -> HF[2]
//   - SP[n] (n>=3)  -> HF[n+1] (normal tokens shifted by 1 due to <pad> insertion)
func (t *Tokenizer) spIndexToHFID(spIndex int32) int32 {
	switch spIndex {
	case 0: // <unk>
		return 3
	case 1: // <s>
		return 0
	case 2: // </s>
		return 2
	default: // normal tokens: shift by 1
		return spIndex + 1
	}
}

// Close releases tokenizer resources.
func (t *Tokenizer) Close() error {
	return nil
}

// VocabSize returns the vocabulary size (HuggingFace XLM-RoBERTa compatible: 250002).
// This is SentencePiece vocab size + 2 (for the inserted <pad> token and the ID shift).
func (t *Tokenizer) VocabSize() int {
	return len(t.idToPiece) + 2
}

// BOSID returns the beginning-of-sentence token ID.
func (t *Tokenizer) BOSID() int32 { return t.bosID }

// PadID returns the padding token ID.
func (t *Tokenizer) PadID() int32 { return t.padID }

// EOSID returns the end-of-sentence token ID.
func (t *Tokenizer) EOSID() int32 { return t.eosID }

// UnkID returns the unknown token ID.
func (t *Tokenizer) UnkID() int32 { return t.unkID }
