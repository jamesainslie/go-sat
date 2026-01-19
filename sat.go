package sat

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"

	"github.com/jamesainslie/go-sat/inference"
	"github.com/jamesainslie/go-sat/tokenizer"
)

// Segmenter detects sentence boundaries using wtpsplit/SaT ONNX models.
// It is safe for concurrent use.
type Segmenter struct {
	tokenizer *tokenizer.Tokenizer
	pool      *inference.Pool
	threshold float32
	logger    *slog.Logger
}

// New creates a Segmenter with the specified model files.
func New(modelPath, tokenizerPath string, opts ...Option) (*Segmenter, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	// Check model file exists
	if _, err := os.Stat(modelPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s", ErrModelNotFound, modelPath)
		}
		return nil, fmt.Errorf("checking model file: %w", err)
	}

	// Load tokenizer
	tok, err := tokenizer.New(tokenizerPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s", ErrTokenizerFailed, tokenizerPath)
		}
		return nil, fmt.Errorf("%w: %w", ErrTokenizerFailed, err)
	}

	// Create session pool
	pool, err := inference.NewPool(modelPath, cfg.poolSize)
	if err != nil {
		_ = tok.Close()
		return nil, fmt.Errorf("%w: %w", ErrInvalidModel, err)
	}

	return &Segmenter{
		tokenizer: tok,
		pool:      pool,
		threshold: cfg.threshold,
		logger:    cfg.logger,
	}, nil
}

// IsComplete returns whether text appears to be a complete sentence.
func (s *Segmenter) IsComplete(ctx context.Context, text string) (complete bool, confidence float32, err error) {
	if text == "" {
		return false, 0.0, nil
	}

	// Tokenize
	tokens := s.tokenizer.Encode(text)
	if len(tokens) == 0 {
		return false, 0.0, nil
	}

	// Prepare model input
	inputIDs := make([]int64, len(tokens))
	attentionMask := make([]int64, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = int64(t.ID)
		attentionMask[i] = 1
	}

	// Acquire session from pool
	session, err := s.pool.Acquire(ctx)
	if err != nil {
		return false, 0, err
	}
	defer s.pool.Release(session)

	// Run inference
	logits, err := session.Infer(ctx, inputIDs, attentionMask)
	if err != nil {
		return false, 0, fmt.Errorf("inference: %w", err)
	}

	// Check last token's boundary probability
	lastLogit := logits[len(logits)-1]
	prob := sigmoid(lastLogit)

	complete = prob > s.threshold
	return complete, prob, nil
}

// Segment splits text into sentences.
func (s *Segmenter) Segment(ctx context.Context, text string) ([]string, error) {
	if text == "" {
		return nil, nil
	}

	// Tokenize
	tokens := s.tokenizer.Encode(text)
	if len(tokens) == 0 {
		return nil, nil
	}

	// Prepare model input
	inputIDs := make([]int64, len(tokens))
	attentionMask := make([]int64, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = int64(t.ID)
		attentionMask[i] = 1
	}

	// Acquire session from pool
	session, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer s.pool.Release(session)

	// Run inference
	logits, err := session.Infer(ctx, inputIDs, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	// Find boundaries using token byte offsets
	var boundaries []int
	for i, logit := range logits {
		if sigmoid(logit) > s.threshold {
			// Map token end position to character position
			if i < len(tokens) {
				boundaries = append(boundaries, tokens[i].End)
			}
		}
	}

	// Split text at boundaries
	if len(boundaries) == 0 {
		return []string{text}, nil
	}

	var sentences []string
	start := 0
	for _, end := range boundaries {
		if end > start && end <= len(text) {
			sentences = append(sentences, text[start:end])
			start = end
		}
	}
	if start < len(text) {
		sentences = append(sentences, text[start:])
	}

	return sentences, nil
}

// Close releases all resources.
func (s *Segmenter) Close() error {
	var errs []error

	if s.pool != nil {
		if err := s.pool.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if s.tokenizer != nil {
		if err := s.tokenizer.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}
