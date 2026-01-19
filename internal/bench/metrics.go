package bench

// Config holds evaluation parameters.
type Config struct {
	Threshold       float32
	Tolerance       int     // character match tolerance
	PrecisionWeight float64
	RecallWeight    float64
}

// DefaultConfig returns default evaluation configuration.
func DefaultConfig() Config {
	return Config{
		Threshold:       0.025,
		Tolerance:       3,
		PrecisionWeight: 1.0,
		RecallWeight:    1.0,
	}
}

// Metrics holds evaluation results.
type Metrics struct {
	TruePositives  int
	FalsePositives int
	FalseNegatives int
	Precision      float64
	Recall         float64
	F1             float64
	WeightedScore  float64
}

// Evaluate compares predicted boundaries against ground truth.
// Uses greedy left-to-right matching within tolerance.
func Evaluate(predicted, truth []int, cfg Config) Metrics {
	matched := make([]bool, len(truth))
	tp := 0

	for _, p := range predicted {
		for i, t := range truth {
			if matched[i] {
				continue
			}
			diff := p - t
			if diff < 0 {
				diff = -diff
			}
			if diff <= cfg.Tolerance {
				matched[i] = true
				tp++
				break
			}
		}
	}

	fp := len(predicted) - tp
	fn := len(truth) - tp

	m := Metrics{
		TruePositives:  tp,
		FalsePositives: fp,
		FalseNegatives: fn,
	}

	if tp+fp > 0 {
		m.Precision = float64(tp) / float64(tp+fp)
	}
	if tp+fn > 0 {
		m.Recall = float64(tp) / float64(tp+fn)
	}
	if m.Precision+m.Recall > 0 {
		m.F1 = 2 * m.Precision * m.Recall / (m.Precision + m.Recall)
	}

	wp := cfg.PrecisionWeight
	wr := cfg.RecallWeight
	if wp+wr > 0 {
		m.WeightedScore = (wp*m.Precision + wr*m.Recall) / (wp + wr)
	}

	return m
}
