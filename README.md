# go-sat

Pure Go library for sentence boundary detection using [wtpsplit/SaT](https://github.com/segment-any-text/wtpsplit) ONNX models.

## Installation

```bash
go get github.com/jamesainslie/go-sat
```

**Note:** Requires ONNX Runtime shared library. See [onnxruntime_go](https://github.com/yalue/onnxruntime_go) for installation instructions.

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    sat "github.com/jamesainslie/go-sat"
)

func main() {
    seg, err := sat.New("model_optimized.onnx", "sentencepiece.bpe.model")
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

## Model Files

Download from HuggingFace:

```bash
# Tokenizer
curl -L -o sentencepiece.bpe.model \
  "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model"

# Model (choose one)
# sat-1l-sm (fast, ~400MB)
curl -L -o model_optimized.onnx \
  "https://huggingface.co/segment-any-text/sat-1l-sm/resolve/main/model_optimized.onnx"
```

## Options

```go
seg, err := sat.New(modelPath, tokenizerPath,
    sat.WithThreshold(0.025),     // Boundary detection threshold
    sat.WithPoolSize(4),          // ONNX session pool size
    sat.WithLogger(slog.Default()), // Custom logger
)
```

## CLI Tool

```bash
go install github.com/jamesainslie/go-sat/cmd/sat-cli@latest

sat-cli -model model.onnx -tokenizer tokenizer.model "Hello world."
sat-cli -model model.onnx -tokenizer tokenizer.model -mode segment "Hello. World."
```

## License

MIT
