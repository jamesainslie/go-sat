package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	sat "github.com/jamesainslie/go-sat"
)

func main() {
	modelPath := flag.String("model", "", "Path to ONNX model file")
	tokenizerPath := flag.String("tokenizer", "", "Path to SentencePiece model file")
	threshold := flag.Float64("threshold", 0.025, "Boundary detection threshold")
	mode := flag.String("mode", "complete", "Mode: complete or segment")

	flag.Parse()

	if *modelPath == "" || *tokenizerPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: sat-cli -model MODEL -tokenizer TOKENIZER [OPTIONS] TEXT")
		flag.PrintDefaults()
		os.Exit(1)
	}

	text := strings.Join(flag.Args(), " ")
	if text == "" {
		fmt.Fprintln(os.Stderr, "Error: no text provided")
		os.Exit(1)
	}

	seg, err := sat.New(*modelPath, *tokenizerPath, sat.WithThreshold(float32(*threshold)))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating segmenter: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = seg.Close() }() // Cleanup error ignored in CLI

	ctx := context.Background()

	switch *mode {
	case "complete":
		complete, confidence, err := seg.IsComplete(ctx, text)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Text: %q\n", text)
		fmt.Printf("Complete: %v\n", complete)
		fmt.Printf("Confidence: %.4f\n", confidence)

	case "segment":
		sentences, err := seg.Segment(ctx, text)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Text: %q\n", text)
		fmt.Printf("Sentences (%d):\n", len(sentences))
		for i, s := range sentences {
			fmt.Printf("  %d: %q\n", i+1, s)
		}

	default:
		fmt.Fprintf(os.Stderr, "Unknown mode: %s\n", *mode)
		os.Exit(1)
	}
}
