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

const (
	// maxSeqLen is the maximum sequence length supported by the model.
	// The model supports positions 0-513, so max is 514 tokens.
	// We use 512 to leave margin for safety.
	maxSeqLen = 512

	// chunkOverlap is the number of overlapping tokens between chunks.
	// This ensures boundary detection works properly at chunk boundaries.
	chunkOverlap = 64
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

	// Get logits for all tokens, handling chunking if needed
	logits, err := s.getLogits(ctx, tokens)
	if err != nil {
		return false, 0, err
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

	// Get logits for all tokens, handling chunking if needed
	logits, err := s.getLogits(ctx, tokens)
	if err != nil {
		return nil, err
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

// SegmentWithBoundaries splits text into sentences and returns boundary positions.
// Boundaries are character offsets where each sentence ends in the original text.
func (s *Segmenter) SegmentWithBoundaries(ctx context.Context, text string) (sentences []string, boundaries []int, err error) {
	if text == "" {
		return nil, nil, nil
	}

	// Tokenize
	tokens := s.tokenizer.Encode(text)
	if len(tokens) == 0 {
		return nil, nil, nil
	}

	// Get logits for all tokens, handling chunking if needed
	logits, err := s.getLogits(ctx, tokens)
	if err != nil {
		return nil, nil, err
	}

	// Find boundaries using token byte offsets
	for i, logit := range logits {
		if sigmoid(logit) > s.threshold {
			if i < len(tokens) {
				boundaries = append(boundaries, tokens[i].End)
			}
		}
	}

	// Split text at boundaries
	if len(boundaries) == 0 {
		return []string{text}, []int{len(text)}, nil
	}

	start := 0
	for _, end := range boundaries {
		if end > start && end <= len(text) {
			sentences = append(sentences, text[start:end])
			start = end
		}
	}
	if start < len(text) {
		sentences = append(sentences, text[start:])
		boundaries = append(boundaries, len(text))
	}

	return sentences, boundaries, nil
}

// getLogits returns logits for all tokens, chunking if necessary.
func (s *Segmenter) getLogits(ctx context.Context, tokens []tokenizer.TokenInfo) ([]float32, error) {
	// Acquire session from pool
	session, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer s.pool.Release(session)

	// If sequence fits in one chunk, process directly
	if len(tokens) <= maxSeqLen {
		return s.inferChunk(ctx, session, tokens)
	}

	// Process in overlapping chunks
	logits := make([]float32, len(tokens))
	counts := make([]int, len(tokens)) // Track how many times each position was processed

	stride := maxSeqLen - chunkOverlap
	for start := 0; start < len(tokens); start += stride {
		end := start + maxSeqLen
		if end > len(tokens) {
			end = len(tokens)
		}

		chunk := tokens[start:end]
		chunkLogits, err := s.inferChunk(ctx, session, chunk)
		if err != nil {
			return nil, err
		}

		// Accumulate logits (for averaging in overlap regions)
		for i, logit := range chunkLogits {
			logits[start+i] += logit
			counts[start+i]++
		}

		// Stop if we've reached the end
		if end >= len(tokens) {
			break
		}
	}

	// Average logits in overlapping regions
	for i := range logits {
		if counts[i] > 1 {
			logits[i] /= float32(counts[i])
		}
	}

	return logits, nil
}

// inferChunk runs inference on a single chunk of tokens.
func (s *Segmenter) inferChunk(ctx context.Context, session *inference.Session, tokens []tokenizer.TokenInfo) ([]float32, error) {
	inputIDs := make([]int64, len(tokens))
	attentionMask := make([]int64, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = int64(t.ID)
		attentionMask[i] = 1
	}

	return session.Infer(ctx, inputIDs, attentionMask)
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
