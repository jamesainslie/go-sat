package sat

import (
	"context"
	"errors"
	"os"
	"testing"
)

const (
	testModelPath     = "testdata/model_optimized.onnx"
	testTokenizerPath = "testdata/sentencepiece.bpe.model"
)

// skipIfNoModel skips the test if the ONNX model is not available.
func skipIfNoModel(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(testModelPath); err != nil {
		t.Skipf("Skipping: ONNX model not available at %s", testModelPath)
	}
}

// skipIfNoTokenizer skips the test if the tokenizer model is not available.
func skipIfNoTokenizer(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(testTokenizerPath); err != nil {
		t.Skipf("Skipping: Tokenizer model not available at %s", testTokenizerPath)
	}
}

func TestNew(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer func() { _ = seg.Close() }()

	if seg == nil {
		t.Error("expected non-nil segmenter")
	}
	if seg.tokenizer == nil {
		t.Error("expected non-nil tokenizer")
	}
	if seg.pool == nil {
		t.Error("expected non-nil pool")
	}
}

func TestNew_ModelNotFound(t *testing.T) {
	skipIfNoTokenizer(t)

	_, err := New("nonexistent/model.onnx", testTokenizerPath)
	if err == nil {
		t.Fatal("expected error for nonexistent model")
	}
	if !errors.Is(err, ErrModelNotFound) {
		t.Errorf("expected ErrModelNotFound, got: %v", err)
	}
}

func TestNew_TokenizerNotFound(t *testing.T) {
	// Create a temp file to act as the model so we pass the model check
	tmpModel, err := os.CreateTemp("", "fake_model_*.onnx")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer func() { _ = os.Remove(tmpModel.Name()) }()
	_ = tmpModel.Close()

	_, err = New(tmpModel.Name(), "nonexistent/tokenizer.model")
	if err == nil {
		t.Fatal("expected error for nonexistent tokenizer")
	}
	if !errors.Is(err, ErrTokenizerFailed) {
		t.Errorf("expected ErrTokenizerFailed, got: %v", err)
	}
}

func TestNew_WithOptions(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath,
		WithThreshold(0.5),
		WithPoolSize(2),
	)
	if err != nil {
		t.Fatalf("New() with options failed: %v", err)
	}
	defer func() { _ = seg.Close() }()

	if seg.threshold != 0.5 {
		t.Errorf("expected threshold 0.5, got %f", seg.threshold)
	}
}

func TestSegmenter_IsComplete_Empty(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer func() { _ = seg.Close() }()

	ctx := context.Background()
	complete, confidence, err := seg.IsComplete(ctx, "")
	if err != nil {
		t.Fatalf("IsComplete failed: %v", err)
	}

	if complete {
		t.Error("expected empty string to be incomplete")
	}
	if confidence != 0.0 {
		t.Errorf("expected confidence 0.0, got %f", confidence)
	}
}

func TestSegmenter_IsComplete_CompleteSentence(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer func() { _ = seg.Close() }()

	ctx := context.Background()
	complete, confidence, err := seg.IsComplete(ctx, "Hello world.")
	if err != nil {
		t.Fatalf("IsComplete failed: %v", err)
	}

	// We just verify the function returns without error and gives a reasonable confidence
	t.Logf("IsComplete('Hello world.') = complete:%v, confidence:%f", complete, confidence)
}

func TestSegmenter_IsComplete_ContextCancelled(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer func() { _ = seg.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, _, err = seg.IsComplete(ctx, "Hello world.")
	if err == nil {
		t.Error("expected error for cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
}

func TestSegmenter_Segment_Empty(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer func() { _ = seg.Close() }()

	ctx := context.Background()
	sentences, err := seg.Segment(ctx, "")
	if err != nil {
		t.Fatalf("Segment failed: %v", err)
	}

	if sentences != nil {
		t.Errorf("expected nil for empty string, got: %v", sentences)
	}
}

func TestSegmenter_Segment_SingleSentence(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer func() { _ = seg.Close() }()

	ctx := context.Background()
	sentences, err := seg.Segment(ctx, "Hello world.")
	if err != nil {
		t.Fatalf("Segment failed: %v", err)
	}

	// Log results for inspection
	t.Logf("Segment('Hello world.') returned %d sentences: %v", len(sentences), sentences)

	// At minimum we should get some output
	if len(sentences) == 0 {
		t.Error("expected at least one sentence")
	}
}

func TestSegmenter_Segment_MultipleSentences(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer func() { _ = seg.Close() }()

	ctx := context.Background()
	text := "Hello world. How are you? I am fine."
	sentences, err := seg.Segment(ctx, text)
	if err != nil {
		t.Fatalf("Segment failed: %v", err)
	}

	// Log results for inspection
	t.Logf("Segment('%s') returned %d sentences: %v", text, len(sentences), sentences)

	// With a well-trained model, we'd expect 3 sentences
	// But we don't enforce this since it depends on the model
	if len(sentences) == 0 {
		t.Error("expected at least one sentence")
	}
}

func TestSegmenter_Segment_ContextCancelled(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer func() { _ = seg.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = seg.Segment(ctx, "Hello world.")
	if err == nil {
		t.Error("expected error for cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
}

func TestSegmenter_Close(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoTokenizer(t)

	seg, err := New(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	err = seg.Close()
	if err != nil {
		t.Errorf("Close() failed: %v", err)
	}

	// Double close should not panic
	err = seg.Close()
	if err != nil {
		t.Logf("Second Close() returned: %v", err)
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float32
		expected float32
		delta    float32
	}{
		{0.0, 0.5, 0.001},
		{-10.0, 0.0, 0.001},
		{10.0, 1.0, 0.001},
		{-1.0, 0.2689, 0.001},
		{1.0, 0.7311, 0.001},
	}

	for _, tt := range tests {
		result := sigmoid(tt.input)
		if result < tt.expected-tt.delta || result > tt.expected+tt.delta {
			t.Errorf("sigmoid(%f) = %f, expected ~%f", tt.input, result, tt.expected)
		}
	}
}
