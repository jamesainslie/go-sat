# API Reference

This document provides detailed documentation for all public types and functions in the go-sat library.

## Package sat

```go
import sat "github.com/jamesainslie/go-sat"
```

### Types

#### Segmenter

```go
type Segmenter struct {
    // contains filtered or unexported fields
}
```

Segmenter detects sentence boundaries using wtpsplit/SaT ONNX models. It is safe for concurrent use.

A Segmenter manages:

- A tokenizer for converting text to token IDs
- A pool of ONNX sessions for inference
- Configuration (threshold, logger)

#### Option

```go
type Option func(*config)
```

Option configures a Segmenter.

### Functions

#### New

```go
func New(modelPath, tokenizerPath string, opts ...Option) (*Segmenter, error)
```

New creates a Segmenter with the specified model files.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `modelPath` | `string` | Path to the ONNX model file |
| `tokenizerPath` | `string` | Path to the SentencePiece model file |
| `opts` | `...Option` | Configuration options |

**Returns:**

| Type | Description |
|------|-------------|
| `*Segmenter` | The initialized segmenter |
| `error` | Error if initialization fails |

**Errors:**

- `ErrModelNotFound`: The ONNX model file does not exist
- `ErrTokenizerFailed`: The tokenizer model file does not exist or is invalid
- `ErrInvalidModel`: The ONNX model file exists but is malformed

**Example:**

```go
seg, err := sat.New("model.onnx", "tokenizer.model")
if err != nil {
    log.Fatal(err)
}
defer seg.Close()
```

**Example with options:**

```go
seg, err := sat.New("model.onnx", "tokenizer.model",
    sat.WithThreshold(0.05),
    sat.WithPoolSize(2),
    sat.WithLogger(slog.Default()),
)
if err != nil {
    log.Fatal(err)
}
defer seg.Close()
```

### Methods

#### (*Segmenter) IsComplete

```go
func (s *Segmenter) IsComplete(ctx context.Context, text string) (complete bool, confidence float32, err error)
```

IsComplete returns whether text appears to be a complete sentence.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `ctx` | `context.Context` | Context for cancellation |
| `text` | `string` | Text to analyze |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `complete` | `bool` | True if text appears to be a complete sentence |
| `confidence` | `float32` | Boundary probability (0.0 to 1.0) |
| `err` | `error` | Error if inference fails |

**Behavior:**

- Empty string returns `false, 0.0, nil`
- Tokenizes the text and runs inference
- Returns the boundary probability of the last token
- `complete` is true if probability exceeds the configured threshold

**Example:**

```go
ctx := context.Background()
complete, confidence, err := seg.IsComplete(ctx, "Hello world.")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Complete: %v (confidence: %.2f)\n", complete, confidence)
```

**Example with timeout:**

```go
ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
defer cancel()

complete, confidence, err := seg.IsComplete(ctx, "Hello world.")
if errors.Is(err, context.DeadlineExceeded) {
    // Handle timeout
}
```

#### (*Segmenter) Segment

```go
func (s *Segmenter) Segment(ctx context.Context, text string) ([]string, error)
```

Segment splits text into sentences.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `ctx` | `context.Context` | Context for cancellation |
| `text` | `string` | Text to segment |

**Returns:**

| Type | Description |
|------|-------------|
| `[]string` | Slice of sentences |
| `error` | Error if inference fails |

**Behavior:**

- Empty string returns `nil, nil`
- Tokenizes the text and runs inference
- Finds all token positions where boundary probability exceeds threshold
- Splits text at those positions
- Returns at least one segment if text is non-empty

**Example:**

```go
ctx := context.Background()
sentences, err := seg.Segment(ctx, "Hello world. How are you? I am fine.")
if err != nil {
    log.Fatal(err)
}
for i, s := range sentences {
    fmt.Printf("%d: %s\n", i+1, s)
}
```

**Output:**

```
1: Hello world.
2:  How are you?
3:  I am fine.
```

Note: Leading whitespace is preserved in segments after the first.

#### (*Segmenter) Close

```go
func (s *Segmenter) Close() error
```

Close releases all resources including the ONNX session pool and tokenizer.

**Returns:**

| Type | Description |
|------|-------------|
| `error` | Error if cleanup fails; may contain multiple errors joined |

**Behavior:**

- Closes all sessions in the pool
- Releases tokenizer resources
- Safe to call multiple times
- Returns combined errors if multiple resources fail to close

**Example:**

```go
seg, err := sat.New("model.onnx", "tokenizer.model")
if err != nil {
    log.Fatal(err)
}
defer func() {
    if err := seg.Close(); err != nil {
        log.Printf("Warning: close failed: %v", err)
    }
}()
```

### Options

#### WithThreshold

```go
func WithThreshold(t float32) Option
```

WithThreshold sets the boundary detection threshold.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `t` | `float32` | Threshold value (0.0 to 1.0) |

**Default:** 0.025

A token is considered a sentence boundary if its probability exceeds this threshold. Lower values detect more boundaries (more aggressive splitting). Higher values require higher confidence (fewer splits).

**Example:**

```go
// More aggressive splitting
seg, _ := sat.New(modelPath, tokenizerPath, sat.WithThreshold(0.01))

// More conservative splitting
seg, _ := sat.New(modelPath, tokenizerPath, sat.WithThreshold(0.1))
```

#### WithPoolSize

```go
func WithPoolSize(n int) Option
```

WithPoolSize sets the ONNX session pool size.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n` | `int` | Number of sessions (must be > 0) |

**Default:** `runtime.NumCPU()`

The pool size determines maximum concurrent inferences. Values <= 0 are ignored (default used).

**Example:**

```go
// Limit to 2 concurrent inferences (lower memory)
seg, _ := sat.New(modelPath, tokenizerPath, sat.WithPoolSize(2))

// Allow 8 concurrent inferences (higher throughput)
seg, _ := sat.New(modelPath, tokenizerPath, sat.WithPoolSize(8))
```

#### WithLogger

```go
func WithLogger(l *slog.Logger) Option
```

WithLogger sets the logger for the Segmenter.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `l` | `*slog.Logger` | Logger instance (nil uses default) |

**Default:** `slog.Default()`

**Example:**

```go
logger := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
    Level: slog.LevelDebug,
}))
seg, _ := sat.New(modelPath, tokenizerPath, sat.WithLogger(logger))
```

### Errors

#### ErrModelNotFound

```go
var ErrModelNotFound = errors.New("sat: model file not found")
```

ErrModelNotFound indicates the ONNX model file does not exist at the specified path.

#### ErrInvalidModel

```go
var ErrInvalidModel = errors.New("sat: invalid model format")
```

ErrInvalidModel indicates the ONNX model file exists but is malformed or incompatible.

#### ErrTokenizerFailed

```go
var ErrTokenizerFailed = errors.New("sat: tokenizer initialization failed")
```

ErrTokenizerFailed indicates tokenizer initialization failed. This occurs when the SentencePiece model file does not exist or cannot be parsed.

**Error handling example:**

```go
seg, err := sat.New(modelPath, tokenizerPath)
if err != nil {
    switch {
    case errors.Is(err, sat.ErrModelNotFound):
        fmt.Printf("Model file not found: %s\n", modelPath)
    case errors.Is(err, sat.ErrTokenizerFailed):
        fmt.Printf("Tokenizer file not found or invalid: %s\n", tokenizerPath)
    case errors.Is(err, sat.ErrInvalidModel):
        fmt.Println("Model file is invalid")
    default:
        fmt.Printf("Unexpected error: %v\n", err)
    }
    os.Exit(1)
}
```

## Complete Example

```go
package main

import (
    "context"
    "errors"
    "fmt"
    "log"
    "log/slog"
    "os"
    "time"

    sat "github.com/jamesainslie/go-sat"
)

func main() {
    // Create segmenter with options
    seg, err := sat.New(
        "model_optimized.onnx",
        "sentencepiece.bpe.model",
        sat.WithThreshold(0.025),
        sat.WithPoolSize(4),
        sat.WithLogger(slog.Default()),
    )
    if err != nil {
        handleError(err)
    }
    defer seg.Close()

    // Create context with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // Check sentence completeness
    text := "The quick brown fox jumps over the lazy dog."
    complete, confidence, err := seg.IsComplete(ctx, text)
    if err != nil {
        log.Fatalf("IsComplete failed: %v", err)
    }
    fmt.Printf("Text: %q\n", text)
    fmt.Printf("Complete: %v (confidence: %.4f)\n\n", complete, confidence)

    // Segment multiple sentences
    paragraph := "Machine learning is transforming industries. Natural language processing enables new applications. Sentence boundary detection is a fundamental task."
    sentences, err := seg.Segment(ctx, paragraph)
    if err != nil {
        log.Fatalf("Segment failed: %v", err)
    }
    fmt.Printf("Paragraph: %q\n", paragraph)
    fmt.Printf("Found %d sentences:\n", len(sentences))
    for i, s := range sentences {
        fmt.Printf("  %d: %q\n", i+1, s)
    }
}

func handleError(err error) {
    switch {
    case errors.Is(err, sat.ErrModelNotFound):
        fmt.Fprintln(os.Stderr, "Error: ONNX model file not found")
        fmt.Fprintln(os.Stderr, "Download from: https://huggingface.co/segment-any-text/sat-1l-sm")
    case errors.Is(err, sat.ErrTokenizerFailed):
        fmt.Fprintln(os.Stderr, "Error: Tokenizer model file not found or invalid")
        fmt.Fprintln(os.Stderr, "Download from: https://huggingface.co/xlm-roberta-base")
    case errors.Is(err, sat.ErrInvalidModel):
        fmt.Fprintln(os.Stderr, "Error: ONNX model file is invalid")
    default:
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
    }
    os.Exit(1)
}
```

## Thread-Safe Usage

The Segmenter is safe for concurrent use from multiple goroutines:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"

    sat "github.com/jamesainslie/go-sat"
)

func main() {
    seg, err := sat.New("model.onnx", "tokenizer.model", sat.WithPoolSize(4))
    if err != nil {
        log.Fatal(err)
    }
    defer seg.Close()

    texts := []string{
        "Hello world.",
        "How are you?",
        "The weather is nice today.",
        "Go is a great language.",
    }

    var wg sync.WaitGroup
    ctx := context.Background()

    for _, text := range texts {
        wg.Add(1)
        go func(t string) {
            defer wg.Done()
            complete, confidence, err := seg.IsComplete(ctx, t)
            if err != nil {
                log.Printf("Error for %q: %v", t, err)
                return
            }
            fmt.Printf("%q: complete=%v confidence=%.4f\n", t, complete, confidence)
        }(text)
    }

    wg.Wait()
}
```
