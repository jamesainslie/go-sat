package sat

import "errors"

// Sentinel errors for conditions callers may need to handle differently.
var (
	// ErrModelNotFound indicates the model file does not exist.
	ErrModelNotFound = errors.New("sat: model file not found")

	// ErrInvalidModel indicates the model file exists but is malformed.
	ErrInvalidModel = errors.New("sat: invalid model format")

	// ErrTokenizerFailed indicates tokenizer initialization failed.
	ErrTokenizerFailed = errors.New("sat: tokenizer initialization failed")
)
