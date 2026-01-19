package bench

import (
	"context"
	"sort"

	sat "github.com/jamesainslie/go-sat"
)

// SweepResult holds metrics for one threshold value.
type SweepResult struct {
	Threshold float32
	Metrics   Metrics
}

// SweepThresholds generates threshold values from min to max with given step.
func SweepThresholds(min, max, step float32) []float32 {
	var thresholds []float32
	for t := min; t < max; t += step {
		thresholds = append(thresholds, t)
	}
	return thresholds
}

// Sweep evaluates multiple thresholds and returns results sorted by weighted score.
func Sweep(ctx context.Context, talks []*Talk, modelPath, tokenizerPath string, cfg Config, thresholds []float32) ([]SweepResult, error) {
	var results []SweepResult

	for _, threshold := range thresholds {
		seg, err := sat.New(modelPath, tokenizerPath, sat.WithThreshold(threshold))
		if err != nil {
			return nil, err
		}

		// Aggregate metrics across all talks
		var totalTP, totalFP, totalFN int
		for _, talk := range talks {
			cfg.Threshold = threshold
			m, err := EvaluateTalk(ctx, seg, talk, cfg)
			if err != nil {
				_ = seg.Close()
				return nil, err
			}
			totalTP += m.TruePositives
			totalFP += m.FalsePositives
			totalFN += m.FalseNegatives
		}

		_ = seg.Close()

		// Compute aggregate metrics
		agg := Metrics{
			TruePositives:  totalTP,
			FalsePositives: totalFP,
			FalseNegatives: totalFN,
		}
		if totalTP+totalFP > 0 {
			agg.Precision = float64(totalTP) / float64(totalTP+totalFP)
		}
		if totalTP+totalFN > 0 {
			agg.Recall = float64(totalTP) / float64(totalTP+totalFN)
		}
		if agg.Precision+agg.Recall > 0 {
			agg.F1 = 2 * agg.Precision * agg.Recall / (agg.Precision + agg.Recall)
		}
		wp := cfg.PrecisionWeight
		wr := cfg.RecallWeight
		if wp+wr > 0 {
			agg.WeightedScore = (wp*agg.Precision + wr*agg.Recall) / (wp + wr)
		}

		results = append(results, SweepResult{
			Threshold: threshold,
			Metrics:   agg,
		})
	}

	// Sort by weighted score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Metrics.WeightedScore > results[j].Metrics.WeightedScore
	})

	return results, nil
}
