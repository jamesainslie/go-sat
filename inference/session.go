// Package inference provides ONNX Runtime integration for SaT model inference.
package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Sentinel errors for closed resources.
var (
	ErrSessionClosed = errors.New("inference: session is closed")
	ErrPoolClosed    = errors.New("inference: pool is closed")
)

var (
	ortEnvOnce sync.Once
	ortEnvErr  error
)

// initORT initializes ONNX Runtime environment once.
// If ONNXRUNTIME_SHARED_LIBRARY_PATH is set, uses that library path.
func initORT() error {
	ortEnvOnce.Do(func() {
		// Check for custom library path (required on macOS with homebrew)
		if libPath := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
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
		return nil, ErrSessionClosed
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

	// Convert attention_mask to float16 bytes
	// Model expects attention_mask as float16, not int64
	attentionMaskF16 := make([]byte, len(attentionMask)*2)
	for i, v := range attentionMask {
		// float16: 0.0 = 0x0000, 1.0 = 0x3C00 (little-endian: 0x00, 0x3C)
		if v != 0 {
			attentionMaskF16[i*2] = 0x00
			attentionMaskF16[i*2+1] = 0x3C
		}
		// else: already zero
	}
	attentionMaskTensor, err := ort.NewCustomDataTensor(
		ort.NewShape(batchSize, seqLen),
		attentionMaskF16,
		ort.TensorElementDataTypeFloat16,
	)
	if err != nil {
		return nil, fmt.Errorf("creating attention_mask tensor: %w", err)
	}
	defer func() { _ = attentionMaskTensor.Destroy() }()

	// Prepare inputs as Value slice
	inputs := []ort.Value{inputIDsTensor, attentionMaskTensor}

	// Pre-allocate output tensor as float16 with shape [1, seqLen, 1]
	// The model outputs float16 logits
	outputData := make([]byte, seqLen*2) // seqLen * 1 * 2 bytes per float16
	outputTensor, err := ort.NewCustomDataTensor(
		ort.NewShape(batchSize, seqLen, 1),
		outputData,
		ort.TensorElementDataTypeFloat16,
	)
	if err != nil {
		return nil, fmt.Errorf("creating output tensor: %w", err)
	}
	defer func() { _ = outputTensor.Destroy() }()

	outputs := []ort.Value{outputTensor}

	// Run inference
	err = s.session.Run(inputs, outputs)
	if err != nil {
		return nil, fmt.Errorf("running inference: %w", err)
	}

	// Convert float16 bytes to float32 logits
	logits := make([]float32, seqLen)
	for i := int64(0); i < seqLen; i++ {
		// Read float16 (2 bytes, little-endian)
		low := uint16(outputData[i*2])
		high := uint16(outputData[i*2+1])
		f16bits := low | (high << 8)
		logits[i] = float16ToFloat32(f16bits)
	}

	return logits, nil
}

// float16ToFloat32 converts a 16-bit float to 32-bit float.
func float16ToFloat32(f16 uint16) float32 {
	// Extract components
	sign := (f16 >> 15) & 0x1
	exp := (f16 >> 10) & 0x1F
	frac := f16 & 0x3FF

	if exp == 0 {
		if frac == 0 {
			// Zero
			return 0.0
		}
		// Denormalized number
		exp = 1
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x3FF
	} else if exp == 31 {
		// Inf or NaN
		if frac == 0 {
			if sign == 1 {
				return float32(math.Inf(-1))
			}
			return float32(math.Inf(1))
		}
		return float32(math.NaN())
	}

	// Convert to float32 format
	f32exp := uint32(exp-15+127) << 23
	f32frac := uint32(frac) << 13
	f32sign := uint32(sign) << 31

	f32bits := f32sign | f32exp | f32frac
	return math.Float32frombits(f32bits)
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
