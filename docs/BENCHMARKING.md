# Benchmarking Guide

This document explains how to use `sat-bench` to evaluate sentence boundary detection accuracy and interpret the results.

## Quick Start

```bash
sat-bench -model model.onnx -tokenizer tokenizer.model -corpus testdata/ted
```

Output:

```
Loaded 4 talks from testdata/ted

Precision: 0.45  Recall: 0.52  F1: 0.48  Weighted: 0.49
(TP: 376, FP: 454, FN: 348)
```

## Understanding the Metrics

### Core Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Precision | TP / (TP + FP) | Of all predicted boundaries, how many were correct |
| Recall | TP / (TP + FN) | Of all actual boundaries, how many were found |
| F1 | 2 * P * R / (P + R) | Harmonic mean of precision and recall |
| Weighted | (wp * P + wr * R) / (wp + wr) | Configurable blend via `-wp` and `-wr` flags |

### Raw Counts

| Count | Meaning |
|-------|---------|
| TP (True Positives) | Predicted boundaries that matched a ground truth boundary |
| FP (False Positives) | Predicted boundaries with no nearby ground truth boundary |
| FN (False Negatives) | Ground truth boundaries that were not detected |

### Example Interpretation

```
Precision: 0.45  Recall: 0.52  F1: 0.48  Weighted: 0.49
(TP: 376, FP: 454, FN: 348)
```

- 376 sentence boundaries were correctly detected
- 454 predictions were false alarms (no real boundary nearby)
- 348 real boundaries were missed
- 45% of predictions were correct
- 52% of actual boundaries were found

## Boundary Matching

The benchmark compares predicted boundary positions against ground truth boundaries using a tolerance-based matching algorithm.

### Matching Algorithm

1. For each predicted boundary position, scan ground truth boundaries left-to-right
2. If a ground truth boundary is within the tolerance distance and unmatched, mark it as matched (TP)
3. If no match is found, count as FP
4. Unmatched ground truth boundaries count as FN

### Tolerance

The `-tolerance` flag controls how close a prediction must be to count as a match. Default is 3 characters.

A tolerance of 3 means a predicted boundary at position 100 will match ground truth at positions 97-103. This accounts for:

- Whitespace normalization differences
- Minor tokenization variations
- Trailing punctuation handling

## Interpreting Results

### Score Guidelines

| F1 Score | Quality |
|----------|---------|
| > 0.85 | Excellent - suitable for production use |
| 0.70-0.85 | Good - acceptable for most applications |
| 0.50-0.70 | Fair - may need threshold tuning or post-processing |
| < 0.50 | Poor - investigate corpus mismatch or model issues |

### Why Scores May Be Lower Than Expected

**Transcript conventions differ from training data:**
- TED transcripts may use punctuation differently than the model's training corpus
- Spoken language has different sentence structures than written text
- Transcripts may preserve speech disfluencies

**Ground truth ambiguity:**
- Some sentence boundaries are subjective (especially with spoken language)
- Run-on sentences in transcripts
- Sentence fragments common in speech

**Domain mismatch:**
- The model may perform differently on technical vs. conversational content
- Speaking pace and style affect natural boundary positions

### Precision vs. Recall Trade-off

| Priority | When to Use | Flag Settings |
|----------|-------------|---------------|
| High precision | When false positives are costly (legal documents, subtitles) | `-wp 2.0 -wr 1.0` |
| High recall | When missing boundaries is costly (search indexing) | `-wp 1.0 -wr 2.0` |
| Balanced | General use | `-wp 1.0 -wr 1.0` (default) |

## Threshold Sweep

Use `-sweep` to test multiple thresholds and find the optimal value for your use case.

```bash
sat-bench -model model.onnx -tokenizer tokenizer.model -sweep
```

Output:

```
Threshold Sweep Results (wp=1.0, wr=1.0)
--------------------------------------------------
Thresh   Prec     Rec      F1       Weighted
0.010    0.40     0.52     0.45     0.46
0.020    0.41     0.52     0.46     0.47
0.030    0.42     0.52     0.46     0.47
...
0.190    0.44     0.52     0.48     0.48
--------------------------------------------------
Optimal: 0.190 (Weighted: 0.48)
```

### How Sweep Works

1. Tests thresholds from `-sweep-min` to `-sweep-max` in increments of `-sweep-step`
2. Calculates metrics at each threshold
3. Returns results sorted by weighted score
4. Reports the optimal threshold

### Customizing the Sweep

```bash
# Sweep a narrower range with finer steps
sat-bench -model model.onnx -tokenizer tokenizer.model \
    -sweep -sweep-min 0.01 -sweep-max 0.10 -sweep-step 0.005

# Optimize for recall (e.g., search indexing)
sat-bench -model model.onnx -tokenizer tokenizer.model \
    -sweep -wp 1.0 -wr 2.0

# Optimize for precision (e.g., subtitle generation)
sat-bench -model model.onnx -tokenizer tokenizer.model \
    -sweep -wp 2.0 -wr 1.0
```

### Weighted Score Formula

```
WeightedScore = (wp * Precision + wr * Recall) / (wp + wr)
```

When `wp = wr = 1.0`, the weighted score equals the arithmetic mean of precision and recall. This differs from F1, which is the harmonic mean and penalizes imbalanced precision/recall more heavily.

## Model Comparison

Compare multiple models to find the best performer:

```bash
sat-bench -models "sat-1l-sm.onnx,sat-3l.onnx" -tokenizer tokenizer.model -sweep
```

Output:

```
Model Comparison (wp=1.0, wr=1.0)
------------------------------------------------------------
Model                          Thresh   F1       Weighted
sat-1l-sm.onnx                 0.025    0.89     0.91
sat-3l.onnx                    0.030    0.92     0.93
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | (required) | Path to ONNX model file |
| `-tokenizer` | (required) | Path to SentencePiece tokenizer model |
| `-corpus` | `testdata/ted` | Directory containing transcript files |
| `-threshold` | `0.025` | Boundary detection threshold (single evaluation) |
| `-tolerance` | `3` | Character tolerance for boundary matching |
| `-wp` | `1.0` | Precision weight for weighted score |
| `-wr` | `1.0` | Recall weight for weighted score |
| `-sweep` | `false` | Run threshold sweep instead of single evaluation |
| `-sweep-min` | `0.01` | Minimum threshold for sweep |
| `-sweep-max` | `0.20` | Maximum threshold for sweep |
| `-sweep-step` | `0.01` | Step size for sweep |
| `-models` | | Comma-separated model paths for comparison mode |

## Corpus Format

The benchmark expects transcript files in the corpus directory with this format:

```
# Source: https://www.ted.com/talks/example
# Speaker: Speaker Name
# Duration: 19:24

First sentence. Second sentence! Third sentence?
```

Ground truth boundaries are extracted from sentence-ending punctuation (`.`, `!`, `?`) followed by whitespace or end of file.
