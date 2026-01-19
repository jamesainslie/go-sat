package inference

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestNewSession_FileNotFound(t *testing.T) {
	_, err := NewSession("../testdata/nonexistent.onnx")
	if err == nil {
		t.Error("expected error for non-existent file")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected os.ErrNotExist, got: %v", err)
	}
}

func TestNewSession(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	session, err := NewSession(modelPath)
	if err != nil {
		// Skip if ONNX runtime is not available
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewSession failed: %v", err)
	}
	defer func() { _ = session.Close() }()

	if session == nil {
		t.Error("expected non-nil session")
	}
}

func TestSession_Infer(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	session, err := NewSession(modelPath)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewSession failed: %v", err)
	}
	defer func() { _ = session.Close() }()

	// Test inference with sample input
	// These are placeholder token IDs - in real usage they come from the tokenizer
	inputIDs := []int64{0, 35378, 8, 38, 3714, 43033, 5, 2} // <s> Hello , I like cats . </s>
	attentionMask := make([]int64, len(inputIDs))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	ctx := context.Background()
	logits, err := session.Infer(ctx, inputIDs, attentionMask)
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}

	// Should return logits for each input token
	if len(logits) != len(inputIDs) {
		t.Errorf("expected %d logits, got %d", len(inputIDs), len(logits))
	}
}

func TestSession_Infer_ContextCancellation(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	session, err := NewSession(modelPath)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewSession failed: %v", err)
	}
	defer func() { _ = session.Close() }()

	// Create an already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	inputIDs := []int64{0, 35378, 2}
	attentionMask := []int64{1, 1, 1}

	_, err = session.Infer(ctx, inputIDs, attentionMask)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled error, got: %v", err)
	}
}

func TestSession_Infer_ContextTimeout(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	session, err := NewSession(modelPath)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewSession failed: %v", err)
	}
	defer func() { _ = session.Close() }()

	// Create an already-expired context
	ctx, cancel := context.WithTimeout(context.Background(), -time.Second)
	defer cancel()

	inputIDs := []int64{0, 35378, 2}
	attentionMask := []int64{1, 1, 1}

	_, err = session.Infer(ctx, inputIDs, attentionMask)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected context.DeadlineExceeded error, got: %v", err)
	}
}

func TestSession_Close_Idempotent(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	session, err := NewSession(modelPath)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewSession failed: %v", err)
	}

	// First close should succeed
	if err := session.Close(); err != nil {
		t.Errorf("first Close failed: %v", err)
	}

	// Second close should also succeed (idempotent)
	if err := session.Close(); err != nil {
		t.Errorf("second Close failed: %v", err)
	}
}

func TestSession_Infer_AfterClose(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	session, err := NewSession(modelPath)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewSession failed: %v", err)
	}

	// Close the session
	if err := session.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Infer should fail on closed session
	inputIDs := []int64{0, 35378, 2}
	attentionMask := []int64{1, 1, 1}

	_, err = session.Infer(context.Background(), inputIDs, attentionMask)
	if err == nil {
		t.Error("expected error when calling Infer on closed session")
	}
}

// isORTUnavailableError checks if the error indicates ONNX runtime is not available.
func isORTUnavailableError(err error) bool {
	if err == nil {
		return false
	}
	errStr := err.Error()
	// Common ONNX runtime unavailability indicators
	return strings.Contains(errStr, "onnxruntime") ||
		strings.Contains(errStr, "shared library") ||
		strings.Contains(errStr, "dylib") ||
		strings.Contains(errStr, ".so") ||
		strings.Contains(errStr, ".dll") ||
		strings.Contains(errStr, "not found") ||
		strings.Contains(errStr, "cannot open") ||
		strings.Contains(errStr, "initializing ONNX runtime")
}
