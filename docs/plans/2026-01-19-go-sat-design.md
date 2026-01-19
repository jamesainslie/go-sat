# go-sat Design Document

**Date:** 2026-01-19
**Status:** Approved
**Goal:** Pure Go library for sentence boundary detection using wtpsplit/SaT ONNX models

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tokenizer | Pure Go from scratch | Need Unigram algorithm; existing libs only support BPE |
| Concurrency | Session pooling | High throughput, multiple goroutines |
| Model management | Path-only | User manages files; no network dependencies |
| API style | Context-first with errors | Idiomatic Go for potentially slow operations |
| Error handling | Simple wrapped + selective sentinels | Start minimal, add sentinels when callers need to branch |
| Testing | Golden files from Python | Self-contained, reproducible, no Python at test time |

---

## Core API

```go
package sat

import (
    "context"
    "errors"
    "log/slog"
)

// Sentinel errors
var (
    ErrModelNotFound   = errors.New("sat: model file not found")
    ErrInvalidModel    = errors.New("sat: invalid model format")
    ErrTokenizerFailed = errors.New("sat: tokenizer initialization failed")
)

// Segmenter detects sentence boundaries using wtpsplit/SaT ONNX models.
// It is safe for concurrent use - internally manages a pool of ONNX sessions.
type Segmenter struct {
    tokenizer *tokenizer.Tokenizer
    pool      *sessionPool
    threshold float32
    logger    *slog.Logger
}

// New creates a Segmenter with the specified model files.
func New(modelPath, tokenizerPath string, opts ...Option) (*Segmenter, error)

// IsComplete returns whether text appears to be a complete sentence.
func (s *Segmenter) IsComplete(ctx context.Context, text string) (complete bool, confidence float32, err error)

// Segment splits text into sentences.
func (s *Segmenter) Segment(ctx context.Context, text string) ([]string, error)

// Close releases all resources including the ONNX session pool.
func (s *Segmenter) Close() error

// Options
func WithThreshold(t float32) Option      // Default: 0.025
func WithLogger(l *slog.Logger) Option
func WithPoolSize(n int) Option           // Default: runtime.NumCPU()
```

---

## Tokenizer Architecture

```go
package tokenizer

// Tokenizer implements XLM-RoBERTa compatible SentencePiece Unigram tokenization.
type Tokenizer struct {
    pieces    map[string]int32   // token string → ID
    scores    map[string]float32 // token string → log probability
    idToPiece []string           // ID → token string

    // Special token IDs
    bosID int32 // <s> = 0
    padID int32 // <pad> = 1
    eosID int32 // </s> = 2
    unkID int32 // <unk> = 3

    maxTokenLen int // longest token in vocab (optimization)
}

// TokenInfo represents a token with its position in the original text.
type TokenInfo struct {
    ID    int32
    Text  string
    Start int // byte offset in original text
    End   int // byte offset in original text
}

// New loads a tokenizer from a SentencePiece .model file.
func New(modelPath string) (*Tokenizer, error)

// Encode tokenizes text using Viterbi algorithm, returning tokens with offsets.
func (t *Tokenizer) Encode(text string) []TokenInfo

// EncodeIDs returns just the token IDs (for model input).
func (t *Tokenizer) EncodeIDs(text string) []int32

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int32) string
```

**Unigram Algorithm (Viterbi DP):**
1. Prepend `▁` to text (XLM-RoBERTa convention for word boundaries)
2. For each position, find best tokenization using dynamic programming
3. Score = sum of log probabilities; pick maximum
4. Backtrack to recover optimal token sequence
5. Handle unknown chars via `<0xXX>` byte fallback tokens

---

## ONNX Inference & Session Pool

```go
package inference

import ort "github.com/yalue/onnxruntime_go"

// Session wraps an ONNX Runtime session for SaT inference.
type Session struct {
    session *ort.AdvancedSession
    env     *ort.Environment
}

// NewSession creates a single ONNX session from a model file.
func NewSession(modelPath string) (*Session, error)

// Infer runs the model on tokenized input, returns per-token logits.
func (s *Session) Infer(ctx context.Context, inputIDs, attentionMask []int64) ([]float32, error)

// Close releases ONNX resources.
func (s *Session) Close() error

// Pool manages a pool of ONNX sessions for concurrent inference.
type Pool struct {
    sessions chan *Session
    modelPath string
    size     int
}

// NewPool creates a pool of n ONNX sessions.
func NewPool(modelPath string, size int) (*Pool, error)

// Acquire gets a session from the pool, blocking if none available.
// Respects context cancellation.
func (p *Pool) Acquire(ctx context.Context) (*Session, error)

// Release returns a session to the pool.
func (p *Pool) Release(s *Session)

// Close closes all sessions in the pool.
func (p *Pool) Close() error
```

**Model I/O:**
- Input: `input_ids` [1, seq_len] int64, `attention_mask` [1, seq_len] int64
- Output: `logits` [1, seq_len, 1] float32 (boundary probability per token)

**Post-processing:**
- Apply sigmoid: `prob = 1 / (1 + exp(-logit))`
- Map token probabilities to character positions using `TokenInfo.End`

---

## Core Logic

```go
// IsComplete implementation
func (s *Segmenter) IsComplete(ctx context.Context, text string) (bool, float32, error) {
    if text == "" {
        return false, 0.0, nil
    }

    // 1. Tokenize
    tokens := s.tokenizer.Encode(text)

    // 2. Prepare model input
    inputIDs := make([]int64, len(tokens))
    attentionMask := make([]int64, len(tokens))
    for i, t := range tokens {
        inputIDs[i] = int64(t.ID)
        attentionMask[i] = 1
    }

    // 3. Acquire session from pool
    session, err := s.pool.Acquire(ctx)
    if err != nil {
        return false, 0, err
    }
    defer s.pool.Release(session)

    // 4. Run inference
    logits, err := session.Infer(ctx, inputIDs, attentionMask)
    if err != nil {
        return false, 0, fmt.Errorf("inference: %w", err)
    }

    // 5. Check last token's boundary probability
    lastProb := sigmoid(logits[len(logits)-1])
    complete := lastProb > s.threshold

    return complete, lastProb, nil
}

// Segment finds all boundaries, splits text accordingly
func (s *Segmenter) Segment(ctx context.Context, text string) ([]string, error) {
    // Similar flow, but:
    // 1. Map token probs → character positions
    // 2. Find all positions where prob > threshold
    // 3. Split text at those boundaries
    // 4. Return slice of sentences
}
```

---

## Directory Structure

```
go-sat/
├── sat.go                 # Main Segmenter API, Options
├── sat_test.go            # Integration tests
├── errors.go              # Sentinel errors
├── doc.go                 # Package documentation
│
├── tokenizer/
│   ├── tokenizer.go       # Tokenizer struct, Encode, Decode
│   ├── tokenizer_test.go
│   ├── unigram.go         # Viterbi DP algorithm
│   ├── model.go           # Protobuf loading
│   └── normalize.go       # Text preprocessing (▁ handling)
│
├── inference/
│   ├── session.go         # Single ONNX session wrapper
│   ├── session_test.go
│   ├── pool.go            # Session pool
│   └── pool_test.go
│
├── internal/
│   └── proto/
│       └── sentencepiece_model.pb.go  # Generated from .proto
│
├── testdata/
│   ├── tokenizer_golden.json    # Expected tokenizer outputs
│   ├── segment_golden.json      # Expected segmentation outputs
│   └── README.md                # How to regenerate golden files
│
├── scripts/
│   └── generate_golden.py       # Python script to create test data
│
└── cmd/
    └── sat-cli/
        └── main.go              # CLI for testing/debugging
```

---

## Testing Strategy

**Testing Layers:**

```go
// 1. Tokenizer unit tests (testdata/tokenizer_golden.json)
func TestTokenizer_Encode(t *testing.T) {
    tok, _ := tokenizer.New("testdata/sentencepiece.bpe.model")
    golden := loadGolden("testdata/tokenizer_golden.json")

    for _, tc := range golden {
        got := tok.Encode(tc.Input)
        assert.Equal(t, tc.ExpectedIDs, toIDs(got))
        assert.Equal(t, tc.ExpectedOffsets, toOffsets(got))
    }
}

// 2. Inference unit tests (mock tokenizer, real ONNX)
// 3. Integration tests (full pipeline)
// 4. Benchmark tests for latency requirements (<50ms)
```

**Golden file generation (`scripts/generate_golden.py`):**
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

test_cases = ["Hello world.", "I want to", "Thank you very much."]
# Output: JSON with input, token_ids, offsets for each case
```

---

## Implementation Order

| Phase | Component | Validation |
|-------|-----------|------------|
| 1 | Protobuf loading (`internal/proto/`, `tokenizer/model.go`) | Model parses without error |
| 2 | Unigram tokenizer (`tokenizer/`) | Matches Python golden files |
| 3 | ONNX session wrapper (`inference/session.go`) | Runs inference on test input |
| 4 | Session pool (`inference/pool.go`) | Concurrent access works |
| 5 | Segmenter API (`sat.go`) | IsComplete/Segment match Python |
| 6 | CLI tool (`cmd/sat-cli/`) | Manual testing |

---

## Success Criteria

1. `go test ./...` passes
2. Tokenizer output matches Python `AutoTokenizer.from_pretrained("xlm-roberta-base")`
3. `IsComplete()` results match Python wtpsplit for test cases
4. Inference latency < 50ms for typical utterances (< 50 tokens)
5. No CGO dependencies except onnxruntime (required)

---

## Resources

- [SentencePiece Unigram Paper](https://arxiv.org/abs/1808.06226) - Section 3.2
- [SentencePiece C++ Implementation](https://github.com/google/sentencepiece/blob/master/src/unigram_model.cc)
- [go-sentencepiece](https://github.com/eliben/go-sentencepiece) - Reference for protobuf handling
- [onnxruntime_go](https://github.com/yalue/onnxruntime_go) - ONNX Runtime bindings
- [wtpsplit/SaT Models](https://huggingface.co/segment-any-text/sat-1l-sm) - Model files
