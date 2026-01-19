package bench

import "testing"

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
