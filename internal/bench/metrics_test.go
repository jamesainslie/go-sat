package bench

import (
	"context"
	"os"
	"testing"

	sat "github.com/jamesainslie/go-sat"
)

func TestEvaluate(t *testing.T) {
	tests := []struct {
		name      string
		predicted []int
		truth     []int
		tolerance int
		wantTP    int
		wantFP    int
		wantFN    int
	}{
		{
			name:      "perfect match",
			predicted: []int{10, 20, 30},
			truth:     []int{10, 20, 30},
			tolerance: 0,
			wantTP:    3,
			wantFP:    0,
			wantFN:    0,
		},
		{
			name:      "within tolerance",
			predicted: []int{11, 19, 31},
			truth:     []int{10, 20, 30},
			tolerance: 2,
			wantTP:    3,
			wantFP:    0,
			wantFN:    0,
		},
		{
			name:      "false positive",
			predicted: []int{10, 15, 20},
			truth:     []int{10, 20},
			tolerance: 0,
			wantTP:    2,
			wantFP:    1,
			wantFN:    0,
		},
		{
			name:      "false negative",
			predicted: []int{10},
			truth:     []int{10, 20},
			tolerance: 0,
			wantTP:    1,
			wantFP:    0,
			wantFN:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := Config{Tolerance: tt.tolerance}
			got := Evaluate(tt.predicted, tt.truth, cfg)

			if got.TruePositives != tt.wantTP {
				t.Errorf("TruePositives = %d, want %d", got.TruePositives, tt.wantTP)
			}
			if got.FalsePositives != tt.wantFP {
				t.Errorf("FalsePositives = %d, want %d", got.FalsePositives, tt.wantFP)
			}
			if got.FalseNegatives != tt.wantFN {
				t.Errorf("FalseNegatives = %d, want %d", got.FalseNegatives, tt.wantFN)
			}
		})
	}
}

func TestEvaluateTalk(t *testing.T) {
	modelPath := os.Getenv("SAT_MODEL_PATH")
	tokenizerPath := os.Getenv("SAT_TOKENIZER_PATH")
	if modelPath == "" || tokenizerPath == "" {
		t.Skip("SAT_MODEL_PATH and SAT_TOKENIZER_PATH not set")
	}

	seg, err := sat.New(modelPath, tokenizerPath)
	if err != nil {
		t.Fatalf("failed to create segmenter: %v", err)
	}
	defer func() { _ = seg.Close() }()

	talk := &Talk{
		ID:      "test",
		RawText: "Hello world. How are you?",
		Sentences: []Sentence{
			{Text: "Hello world.", Start: 0, End: 12},
			{Text: "How are you?", Start: 13, End: 25},
		},
	}

	cfg := DefaultConfig()
	metrics, err := EvaluateTalk(context.Background(), seg, talk, cfg)
	if err != nil {
		t.Fatalf("EvaluateTalk() error = %v", err)
	}

	// Should get reasonable precision/recall on simple sentences
	if metrics.Precision < 0.5 {
		t.Errorf("Precision = %v, want >= 0.5", metrics.Precision)
	}
}
