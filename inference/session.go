// Package inference provides ONNX Runtime integration for SaT model inference.
package inference

import (
	"context"
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortEnvOnce sync.Once
	ortEnvErr  error
)

// initORT initializes ONNX Runtime environment once.
func initORT() error {
	ortEnvOnce.Do(func() {
		ortEnvErr = ort.InitializeEnvironment()
	})
	return ortEnvErr
}

// Session wraps an ONNX Runtime session for SaT inference.
type Session struct {
	session *ort.DynamicAdvancedSession
	mu      sync.Mutex
	closed  bool
}

// NewSession creates a new ONNX session from a model file.
func NewSession(modelPath string) (*Session, error) {
	// Check file exists
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model file: %w", err)
	}

	if err := initORT(); err != nil {
		return nil, fmt.Errorf("initializing ONNX runtime: %w", err)
	}

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("creating session options: %w", err)
	}
	defer func() { _ = options.Destroy() }() // Cleanup error doesn't affect success

	// Define input/output names (from model inspection)
	inputNames := []string{"input_ids", "attention_mask"}
	outputNames := []string{"logits"}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("creating session: %w", err)
	}

	return &Session{session: session}, nil
}

// Infer runs the model on tokenized input, returns per-token logits.
func (s *Session) Infer(ctx context.Context, inputIDs, attentionMask []int64) ([]float32, error) {
	// Check context before expensive operation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, fmt.Errorf("session is closed")
	}

	batchSize := int64(1)
	seqLen := int64(len(inputIDs))

	// Create input tensors
	inputIDsTensor, err := ort.NewTensor(
		ort.NewShape(batchSize, seqLen),
		inputIDs,
	)
	if err != nil {
		return nil, fmt.Errorf("creating input_ids tensor: %w", err)
	}
	defer func() { _ = inputIDsTensor.Destroy() }()

	attentionMaskTensor, err := ort.NewTensor(
		ort.NewShape(batchSize, seqLen),
		attentionMask,
	)
	if err != nil {
		return nil, fmt.Errorf("creating attention_mask tensor: %w", err)
	}
	defer func() { _ = attentionMaskTensor.Destroy() }()

	// Prepare inputs as Value slice
	inputs := []ort.Value{inputIDsTensor, attentionMaskTensor}

	// Prepare output slice - nil entries will be allocated by Run
	outputs := []ort.Value{nil}

	// Run inference
	err = s.session.Run(inputs, outputs)
	if err != nil {
		return nil, fmt.Errorf("running inference: %w", err)
	}

	// Extract logits from output
	if outputs[0] == nil {
		return nil, fmt.Errorf("no output produced")
	}
	defer func() { _ = outputs[0].Destroy() }()

	logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output tensor type")
	}

	// Copy output data
	logits := make([]float32, seqLen)
	outputData := logitsTensor.GetData()
	copy(logits, outputData[:seqLen])

	return logits, nil
}

// Close releases ONNX resources.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.closed = true
	if s.session != nil {
		return s.session.Destroy()
	}
	return nil
}
