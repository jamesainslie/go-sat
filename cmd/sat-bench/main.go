package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	sat "github.com/jamesainslie/go-sat"
	"github.com/jamesainslie/go-sat/internal/bench"
)

func main() {
	var (
		modelPath     = flag.String("model", "", "Path to ONNX model file (required)")
		tokenizerPath = flag.String("tokenizer", "", "Path to tokenizer model file (required)")
		corpusDir     = flag.String("corpus", "testdata/ted", "Directory containing transcript files")
		threshold     = flag.Float64("threshold", 0.025, "Boundary detection threshold")
		tolerance     = flag.Int("tolerance", 3, "Character tolerance for boundary matching")
		wp            = flag.Float64("wp", 1.0, "Precision weight")
		wr            = flag.Float64("wr", 1.0, "Recall weight")
		sweep         = flag.Bool("sweep", false, "Run threshold sweep")
		sweepMin      = flag.Float64("sweep-min", 0.01, "Sweep minimum threshold")
		sweepMax      = flag.Float64("sweep-max", 0.20, "Sweep maximum threshold")
		sweepStep     = flag.Float64("sweep-step", 0.01, "Sweep step size")
		models        = flag.String("models", "", "Comma-separated model paths for comparison")
	)
	flag.Parse()

	if *modelPath == "" && *models == "" {
		fmt.Fprintln(os.Stderr, "error: -model or -models required")
		flag.Usage()
		os.Exit(1)
	}
	if *tokenizerPath == "" {
		fmt.Fprintln(os.Stderr, "error: -tokenizer required")
		flag.Usage()
		os.Exit(1)
	}

	// Load corpus
	talks, err := bench.LoadCorpus(*corpusDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading corpus: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d talks from %s\n\n", len(talks), *corpusDir)

	cfg := bench.Config{
		Threshold:       float32(*threshold),
		Tolerance:       *tolerance,
		PrecisionWeight: *wp,
		RecallWeight:    *wr,
	}

	ctx := context.Background()

	if *models != "" {
		// Model comparison mode
		modelPaths := strings.Split(*models, ",")
		runModelComparison(ctx, modelPaths, *tokenizerPath, talks, cfg, *sweep, float32(*sweepMin), float32(*sweepMax), float32(*sweepStep))
	} else if *sweep {
		// Single model sweep mode
		runSweep(ctx, *modelPath, *tokenizerPath, talks, cfg, float32(*sweepMin), float32(*sweepMax), float32(*sweepStep))
	} else {
		// Single threshold evaluation
		runSingle(ctx, *modelPath, *tokenizerPath, talks, cfg)
	}
}

func runSingle(ctx context.Context, modelPath, tokenizerPath string, talks []*bench.Talk, cfg bench.Config) {
	seg, err := sat.New(modelPath, tokenizerPath, sat.WithThreshold(cfg.Threshold))
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating segmenter: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = seg.Close() }()

	var totalTP, totalFP, totalFN int
	for _, talk := range talks {
		m, err := bench.EvaluateTalk(ctx, seg, talk, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error evaluating %s: %v\n", talk.ID, err)
			os.Exit(1)
		}
		totalTP += m.TruePositives
		totalFP += m.FalsePositives
		totalFN += m.FalseNegatives
	}

	printMetrics(totalTP, totalFP, totalFN, cfg)
}

func runSweep(ctx context.Context, modelPath, tokenizerPath string, talks []*bench.Talk, cfg bench.Config, min, max, step float32) {
	thresholds := bench.SweepThresholds(min, max, step)

	fmt.Printf("Threshold Sweep Results (wp=%.1f, wr=%.1f)\n", cfg.PrecisionWeight, cfg.RecallWeight)
	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("%-8s %-8s %-8s %-8s %-8s\n", "Thresh", "Prec", "Rec", "F1", "Weighted")

	results, err := bench.Sweep(ctx, talks, modelPath, tokenizerPath, cfg, thresholds)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error during sweep: %v\n", err)
		os.Exit(1)
	}

	// Print sorted by threshold for readability
	for _, t := range thresholds {
		for _, r := range results {
			if r.Threshold == t {
				fmt.Printf("%-8.3f %-8.2f %-8.2f %-8.2f %-8.2f\n",
					r.Threshold, r.Metrics.Precision, r.Metrics.Recall, r.Metrics.F1, r.Metrics.WeightedScore)
				break
			}
		}
	}

	fmt.Println(strings.Repeat("-", 50))
	if len(results) > 0 {
		best := results[0]
		fmt.Printf("Optimal: %.3f (Weighted: %.2f)\n", best.Threshold, best.Metrics.WeightedScore)
	}
}

func runModelComparison(ctx context.Context, modelPaths []string, tokenizerPath string, talks []*bench.Talk, cfg bench.Config, sweep bool, min, max, step float32) {
	fmt.Printf("Model Comparison (wp=%.1f, wr=%.1f)\n", cfg.PrecisionWeight, cfg.RecallWeight)
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("%-30s %-8s %-8s %-8s\n", "Model", "Thresh", "F1", "Weighted")

	for _, modelPath := range modelPaths {
		var bestThreshold float32
		var bestMetrics bench.Metrics

		if sweep {
			thresholds := bench.SweepThresholds(min, max, step)
			results, err := bench.Sweep(ctx, talks, modelPath, tokenizerPath, cfg, thresholds)
			if err != nil {
				fmt.Fprintf(os.Stderr, "error with %s: %v\n", modelPath, err)
				continue
			}
			if len(results) > 0 {
				bestThreshold = results[0].Threshold
				bestMetrics = results[0].Metrics
			}
		} else {
			seg, err := sat.New(modelPath, tokenizerPath, sat.WithThreshold(cfg.Threshold))
			if err != nil {
				fmt.Fprintf(os.Stderr, "error with %s: %v\n", modelPath, err)
				continue
			}
			var totalTP, totalFP, totalFN int
			for _, talk := range talks {
				m, _ := bench.EvaluateTalk(ctx, seg, talk, cfg)
				totalTP += m.TruePositives
				totalFP += m.FalsePositives
				totalFN += m.FalseNegatives
			}
			_ = seg.Close()

			bestThreshold = cfg.Threshold
			bestMetrics = computeMetrics(totalTP, totalFP, totalFN, cfg)
		}

		fmt.Printf("%-30s %-8.3f %-8.2f %-8.2f\n", modelPath, bestThreshold, bestMetrics.F1, bestMetrics.WeightedScore)
	}
}

func printMetrics(tp, fp, fn int, cfg bench.Config) {
	m := computeMetrics(tp, fp, fn, cfg)
	fmt.Printf("Precision: %.2f  Recall: %.2f  F1: %.2f  Weighted: %.2f\n",
		m.Precision, m.Recall, m.F1, m.WeightedScore)
	fmt.Printf("(TP: %d, FP: %d, FN: %d)\n", tp, fp, fn)
}

func computeMetrics(tp, fp, fn int, cfg bench.Config) bench.Metrics {
	m := bench.Metrics{
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
	if cfg.PrecisionWeight+cfg.RecallWeight > 0 {
		m.WeightedScore = (cfg.PrecisionWeight*m.Precision + cfg.RecallWeight*m.Recall) / (cfg.PrecisionWeight + cfg.RecallWeight)
	}
	return m
}
