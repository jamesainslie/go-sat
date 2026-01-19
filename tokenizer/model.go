package tokenizer

import (
	"fmt"
	"os"

	"google.golang.org/protobuf/proto"

	pb "github.com/jamesainslie/go-sat/internal/proto"
)

// Piece represents a vocabulary piece from the model.
type Piece struct {
	Piece string
	Score float32
	Type  pb.ModelProto_SentencePiece_Type
}

// Model represents a loaded SentencePiece model.
type Model struct {
	Pieces         []Piece
	TrainerSpec    *pb.TrainerSpec
	NormalizerSpec *pb.NormalizerSpec
}

// LoadModel loads a SentencePiece model from a .model file.
func LoadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading model file: %w", err)
	}

	var modelProto pb.ModelProto
	if err := proto.Unmarshal(data, &modelProto); err != nil {
		return nil, fmt.Errorf("parsing protobuf: %w", err)
	}

	pieces := make([]Piece, len(modelProto.Pieces))
	for i, p := range modelProto.Pieces {
		pieces[i] = Piece{
			Piece: p.GetPiece(),
			Score: p.GetScore(),
			Type:  p.GetType(),
		}
	}

	return &Model{
		Pieces:         pieces,
		TrainerSpec:    modelProto.TrainerSpec,
		NormalizerSpec: modelProto.NormalizerSpec,
	}, nil
}
