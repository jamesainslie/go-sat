# Benchmark Suite Design

## Overview

A benchmarking suite (`cmd/sat-bench`) that measures sentence boundary detection accuracy against curated TED Talk transcripts with known sentence boundaries.

## Goals

1. **Optimal threshold discovery** - Find the best threshold value for general use
2. **Domain-specific thresholds** - Support different weights for different use cases
3. **Model comparison** - Compare different SaT model variants to find the best performer

## Non-Goals (YAGNI)

- Real-time transcript fetching
- Visual report generation
- Multi-annotator workflows
- Web UI

## Directory Structure

```
cmd/sat-bench/
    main.go              # CLI entry point
testdata/
    ted/
        README.md        # Attribution, source links
        ken_robinson.txt # Talk transcript
        brene_brown.txt
        ...
internal/bench/
    corpus.go            # Load/parse transcripts
    metrics.go           # Precision, recall, F-score
    sweep.go             # Threshold sweep logic
```

## Transcript Format

Simple, human-readable text files with metadata headers:

```
# Source: https://www.ted.com/talks/sir_ken_robinson_do_schools_kill_creativity
# Speaker: Sir Ken Robinson
# Duration: 19:24

Good morning. How are you? It's been great, hasn't it? I've been blown away by the whole thing. In fact, I'm leaving.
```

### Ground Truth Extraction

- Sentence boundaries = positions of `.` `?` `!` followed by whitespace or EOF
- Ignore abbreviations via simple heuristics (Mr., Dr., U.S., etc.)
- Each transcript becomes a `[]Sentence` with `Text` and `StartOffset`/`EndOffset`

## Test Corpus

Six TED Talks selected for diverse speaking styles (~100 minutes total):

| Talk | Speaker | Why |
|------|---------|-----|
| Do Schools Kill Creativity? | Ken Robinson | Fast, witty, British accent |
| The Power of Vulnerability | Brené Brown | Conversational, emotional pauses |
| How Great Leaders Inspire | Simon Sinek | Slow, deliberate, repetitive structure |
| My Stroke of Insight | Jill Bolte Taylor | Technical + emotional, varied pacing |
| The Danger of a Single Story | Chimamanda Adichie | Nigerian accent, literary style |
| Your Body Language Shapes You | Amy Cuddy | Academic but accessible |

## Metrics

### Core Metrics

```go
type Metrics struct {
    TruePositives  int     // Correctly detected boundaries
    FalsePositives int     // Detected boundary where none exists
    FalseNegatives int     // Missed real boundary
    Precision      float64 // TP / (TP + FP)
    Recall         float64 // TP / (TP + FN)
    F1             float64 // Harmonic mean
    WeightedScore  float64 // Custom weighted score
}
```

### Boundary Matching

- A predicted boundary "matches" a ground truth boundary if within N characters (default: 3)
- Tolerance handles whitespace normalization differences
- Each ground truth boundary can only match one prediction (greedy left-to-right)

### Weighted Score Formula

```
WeightedScore = (w_p * Precision + w_r * Recall) / (w_p + w_r)
```

Where `w_p` and `w_r` are user-provided weights (default: 1.0 each = balanced F1).

### Aggregation

- Per-talk metrics calculated independently
- Aggregate = micro-average (sum all TP/FP/FN across talks, then compute ratios)
- Micro-average weights longer talks proportionally

## CLI Interface

### Basic Usage

```bash
# Single threshold evaluation
sat-bench -model model.onnx -tokenizer tokenizer.model
# Output: Precision: 0.92  Recall: 0.87  F1: 0.89  Weighted: 0.89

# Custom threshold
sat-bench -model model.onnx -tokenizer tokenizer.model -threshold 0.05

# Custom weights (recall 2x more important)
sat-bench -model model.onnx -tokenizer tokenizer.model -wp 1.0 -wr 2.0

# Threshold sweep to find optimal
sat-bench -model model.onnx -tokenizer tokenizer.model -sweep
```

### Sweep Output

```
Threshold Sweep Results (wp=1.0, wr=2.0)
----------------------------------------
Thresh  Prec    Rec     F1      Weighted
0.010   0.71    0.98    0.82    0.89
0.025   0.85    0.94    0.89    0.91
0.050   0.92    0.87    0.89    0.89
0.100   0.96    0.72    0.82    0.80
----------------------------------------
Optimal: 0.025 (Weighted: 0.91)
```

### Model Comparison

```bash
sat-bench -models "sat-1l-sm.onnx,sat-3l.onnx" -tokenizer tokenizer.model -sweep
# Runs sweep for each model, shows comparison table
```

## Internal Package Structure

### corpus.go

```go
// Sentence represents a ground-truth sentence with offsets
type Sentence struct {
    Text  string
    Start int
    End   int
}

// Talk represents a loaded transcript
type Talk struct {
    ID        string     // filename stem
    Speaker   string     // from header
    Source    string     // TED URL
    RawText   string     // full text
    Sentences []Sentence // parsed ground truth
}

// LoadCorpus loads all .txt files from testdata/ted/
func LoadCorpus(dir string) ([]Talk, error)

// ParseSentences extracts boundaries from raw text
func ParseSentences(text string) []Sentence
```

### metrics.go

```go
// Config holds evaluation parameters
type Config struct {
    Threshold       float32
    Tolerance       int     // character match tolerance (default: 3)
    PrecisionWeight float64
    RecallWeight    float64
}

// Evaluate compares predicted vs ground truth boundaries
func Evaluate(predicted, truth []int, cfg Config) Metrics

// EvaluateTalk runs segmentation and evaluates against ground truth
func EvaluateTalk(seg *sat.Segmenter, talk Talk, cfg Config) (Metrics, error)
```

### sweep.go

```go
// SweepResult holds metrics for one threshold
type SweepResult struct {
    Threshold float32
    Metrics   Metrics
}

// Sweep tests multiple thresholds, returns sorted by weighted score
func Sweep(talks []Talk, modelPath, tokPath string, cfg Config) ([]SweepResult, error)
```

## Implementation Order

1. Create transcript fixtures (manual curation)
2. Build corpus loader with sentence parsing
3. Build metrics calculation
4. Build CLI with single-threshold mode
5. Add sweep mode
6. Add model comparison mode

## Out of Scope

- Automated transcript downloading
- Visual charts/reports
- Persistent result storage
- Web interface
