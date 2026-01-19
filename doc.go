// Package sat provides sentence boundary detection using wtpsplit/SaT ONNX models.
//
// # Quick Start
//
//	seg, err := sat.New("model.onnx", "tokenizer.model")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer seg.Close()
//
//	complete, confidence, err := seg.IsComplete(ctx, "Hello world.")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Complete: %v (confidence: %.2f)\n", complete, confidence)
//
// # Thread Safety
//
// Segmenter is safe for concurrent use. It manages an internal pool of ONNX
// sessions, configurable via WithPoolSize.
//
// # Model Files
//
// Download from HuggingFace:
//   - Model: https://huggingface.co/segment-any-text/sat-1l-sm/resolve/main/model_optimized.onnx
//   - Tokenizer: https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model
package sat
