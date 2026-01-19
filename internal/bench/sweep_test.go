package bench

import (
	"testing"
)

func TestSweepThresholds(t *testing.T) {
	thresholds := SweepThresholds(0.01, 0.1, 0.02)

	want := []float32{0.01, 0.03, 0.05, 0.07, 0.09}
	if len(thresholds) != len(want) {
		t.Errorf("got %d thresholds, want %d", len(thresholds), len(want))
		t.Logf("got: %v", thresholds)
		return
	}

	for i := range want {
		diff := thresholds[i] - want[i]
		if diff < -0.001 || diff > 0.001 {
			t.Errorf("threshold[%d] = %v, want %v", i, thresholds[i], want[i])
		}
	}
}
