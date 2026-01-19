# go-sat

[![Go Reference](https://pkg.go.dev/badge/github.com/jamesainslie/go-sat.svg)](https://pkg.go.dev/github.com/jamesainslie/go-sat)
[![Go Report Card](https://goreportcard.com/badge/github.com/jamesainslie/go-sat)](https://goreportcard.com/report/github.com/jamesainslie/go-sat)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Pure Go library for sentence boundary detection using [wtpsplit/SaT](https://github.com/segment-any-text/wtpsplit) ONNX models.

## Features

- Sentence boundary detection using neural models
- Sentence completeness checking with confidence scores
- Text segmentation into sentences
- Automatic chunking for long texts (handles sequences > 512 tokens)
- Thread-safe with configurable session pooling
- Pure Go tokenizer (SentencePiece Unigram algorithm)
- Benchmark tool for evaluating model accuracy

## Requirements

- Go 1.23 or later
- ONNX Runtime shared library

## Installation

```bash
go get github.com/jamesainslie/go-sat
```

### ONNX Runtime

The library requires the ONNX Runtime shared library. See the [onnxruntime_go installation guide](https://github.com/yalue/onnxruntime_go#installation) for platform-specific instructions.

**macOS (Homebrew):**

```bash
brew install onnxruntime
export ONNXRUNTIME_SHARED_LIBRARY_PATH="$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"
```

**Linux:**

```bash
# Download from https://github.com/microsoft/onnxruntime/releases
# Extract and set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

## Model Files

Download the required model files from HuggingFace:

```bash
# Tokenizer (XLM-RoBERTa SentencePiece model)
curl -L -o sentencepiece.bpe.model \
  "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model"

# SaT model (sat-1l-sm: fast, suitable for most use cases)
curl -L -o model.onnx \
  "https://huggingface.co/segment-any-text/sat-1l-sm/resolve/main/model.onnx"
```

**Note:** Use `model.onnx`, not `model_optimized.onnx`. The optimized model uses different input tensor types.

## Usage

### Basic Example

```go
package main

import (
    "context"
    "fmt"
    "log"

    sat "github.com/jamesainslie/go-sat"
)

func main() {
    seg, err := sat.New("model.onnx", "sentencepiece.bpe.model")
    if err != nil {
        log.Fatal(err)
    }
    defer seg.Close()

    ctx := context.Background()

    // Check if text is a complete sentence
    complete, confidence, err := seg.IsComplete(ctx, "Hello world.")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Complete: %v (confidence: %.2f)\n", complete, confidence)

    // Split text into sentences
    sentences, err := seg.Segment(ctx, "Hello world. How are you?")
    if err != nil {
        log.Fatal(err)
    }
    for _, s := range sentences {
        fmt.Println(s)
    }
}
```

### Configuration Options

```go
seg, err := sat.New(modelPath, tokenizerPath,
    sat.WithThreshold(0.025),       // Boundary detection threshold (default: 0.025)
    sat.WithPoolSize(4),            // ONNX session pool size (default: runtime.NumCPU())
    sat.WithLogger(slog.Default()), // Custom logger (default: slog.Default())
)
```

## Architecture

### Components

| Component | Description |
|-----------|-------------|
| **Segmenter** | Main API - coordinates tokenization, inference, and boundary detection |
| **Tokenizer** | Pure Go SentencePiece implementation (Unigram algorithm with Viterbi decoding) |
| **Session Pool** | Thread-safe pool of ONNX Runtime sessions for concurrent inference |
| **Inference** | Handles float16 tensor conversion and model execution |

### Data Flow

1. Text input normalized (whitespace handling, SentencePiece prefix)
2. Tokenized using Viterbi dynamic programming algorithm
3. Token IDs remapped from SentencePiece to HuggingFace convention
4. For long texts (>512 tokens), split into overlapping chunks (64 token overlap)
5. ONNX model inference produces per-token boundary logits (float16)
6. Logits converted to float32, averaged in overlap regions
7. Sigmoid applied; positions above threshold mark sentence boundaries

### Model Requirements

The SaT model expects:
- `input_ids`: int64 tensor of token IDs
- `attention_mask`: float16 tensor (1.0 for valid tokens)
- Output: float16 logits with shape [batch, sequence, 1]

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed technical documentation.

## API Reference

| Function | Description |
|----------|-------------|
| `New(modelPath, tokenizerPath string, opts ...Option)` | Create a new Segmenter |
| `(*Segmenter).IsComplete(ctx, text) (bool, float32, error)` | Check if text is a complete sentence |
| `(*Segmenter).Segment(ctx, text) ([]string, error)` | Split text into sentences |
| `(*Segmenter).Close() error` | Release all resources |

See [docs/API.md](docs/API.md) for detailed API documentation with examples.

## CLI Tools

### sat-cli

Command-line tool for testing and debugging:

```bash
go install github.com/jamesainslie/go-sat/cmd/sat-cli@latest

# Check sentence completeness
sat-cli -model model.onnx -tokenizer tokenizer.model "Hello world."

# Segment text into sentences
sat-cli -model model.onnx -tokenizer tokenizer.model -mode segment "Hello. World."
```

### sat-bench

Benchmark tool for evaluating model accuracy against a test corpus:

```bash
go install github.com/jamesainslie/go-sat/cmd/sat-bench@latest

# Evaluate against UD-EWT gold-standard corpus (recommended)
./scripts/fetch-ud-ewt.sh && go run ./scripts/process-ud-ewt.go
sat-bench -model model.onnx -tokenizer tokenizer.model -corpus testdata/ud-ewt-test -tolerance 10 -sweep

# Or evaluate against TED transcripts
sat-bench -model model.onnx -tokenizer tokenizer.model -corpus testdata/ted -threshold 0.025
```

Example output (UD-EWT test set):

```
Precision: 0.87  Recall: 0.80  F1: 0.84  Weighted: 0.84
(TP: 1662, FP: 239, FN: 415)
```

See [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for detailed guidance on interpreting results and corpus formats.

## Building

The project uses [Stave](https://github.com/yaklabco/stave) for build automation:

```bash
# Install stave
go install github.com/yaklabco/stave/cmd/stave@latest

# Build everything (lint, test, build)
stave

# Quick shortcuts
stave b          # Build binaries
stave t          # Run tests
stave l          # Run linter

# Run benchmarks (auto-configures ONNX runtime on macOS)
stave bench:run
stave bench:sweep
```

See `stave --list` for all available targets.

## Performance

- Inference latency: Typically < 50ms for utterances under 50 tokens
- Thread-safe: Multiple goroutines can call methods concurrently
- Session pooling: Configurable pool size to balance memory and throughput

The default pool size equals `runtime.NumCPU()`. For memory-constrained environments, reduce pool size. For high-throughput scenarios, ensure pool size matches expected concurrency.

## Error Handling

The library defines sentinel errors for conditions callers may need to handle:

```go
var (
    ErrModelNotFound   = errors.New("sat: model file not found")
    ErrInvalidModel    = errors.New("sat: invalid model format")
    ErrTokenizerFailed = errors.New("sat: tokenizer initialization failed")
)
```

Example error handling:

```go
seg, err := sat.New(modelPath, tokenizerPath)
if errors.Is(err, sat.ErrModelNotFound) {
    // Handle missing model file
}
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT
