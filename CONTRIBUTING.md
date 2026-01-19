# Contributing to go-sat

## Development Setup

### Prerequisites

- Go 1.23 or later
- ONNX Runtime shared library
- Git

### Clone and Build

```bash
git clone https://github.com/jamesainslie/go-sat.git
cd go-sat
go build ./...
```

### Install ONNX Runtime

**macOS:**

```bash
brew install onnxruntime
```

**Linux:**

Download from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) page. Extract and configure the library path:

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

### Download Test Data

Tests require model files. Download them to `testdata/`:

```bash
cd testdata

# Tokenizer model
curl -L -o sentencepiece.bpe.model \
  "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model"

# ONNX model
curl -L -o model_optimized.onnx \
  "https://huggingface.co/segment-any-text/sat-1l-sm/resolve/main/model_optimized.onnx"
```

## Running Tests

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run specific package tests
go test -v ./tokenizer/...
go test -v ./inference/...

# Run with race detector
go test -race ./...
```

Tests skip automatically when model files are not present. Integration tests that require ONNX inference will be skipped without the model files.

## Code Style

### Formatting

All code must be formatted with `gofmt`:

```bash
gofmt -w .
```

### Linting

Run golangci-lint before submitting:

```bash
golangci-lint run
```

### Guidelines

- Follow standard Go idioms and conventions
- Use `context.Context` as the first parameter for potentially slow operations
- Return errors; do not panic
- Document exported types and functions
- Write tests for new functionality

### Error Handling

- Wrap errors with context using `fmt.Errorf("context: %w", err)`
- Use sentinel errors for conditions callers may need to distinguish
- Check context cancellation in long-running operations

### Concurrency

- Document thread safety guarantees
- Use mutexes for protecting shared state
- Respect context cancellation

## Testing Guidelines

### Test Organization

- Unit tests in `*_test.go` files alongside the code
- Use table-driven tests for multiple cases
- Skip tests gracefully when dependencies are unavailable

### Golden Files

The tokenizer uses golden files generated from Python for validation. To regenerate:

```bash
cd scripts
pip install transformers
python3 generate_golden.py
```

Golden files are stored in `testdata/`. Regenerate them only when intentionally changing tokenizer behavior.

### Test Helpers

```go
// Skip if model files unavailable
func skipIfNoModel(t *testing.T) {
    t.Helper()
    if _, err := os.Stat(testModelPath); err != nil {
        t.Skipf("Skipping: model not available at %s", testModelPath)
    }
}
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make changes with tests
4. Run `go test ./...` and `golangci-lint run`
5. Commit with clear messages
6. Open a pull request against `main`

### Commit Messages

Use clear, descriptive commit messages:

```
Add boundary detection threshold option

- Add WithThreshold option function
- Update Segmenter to use configurable threshold
- Add tests for threshold behavior
```

### PR Description

Include:

- Summary of changes
- Motivation or issue reference
- Testing performed
- Breaking changes (if any)

## Project Structure

```
go-sat/
├── sat.go              # Main Segmenter API
├── options.go          # Configuration options
├── errors.go           # Sentinel errors
├── doc.go              # Package documentation
├── sat_test.go         # Integration tests
├── tokenizer/          # SentencePiece tokenizer
│   ├── tokenizer.go    # Main tokenizer
│   ├── unigram.go      # Viterbi algorithm
│   ├── model.go        # Protobuf loading
│   └── normalize.go    # Text preprocessing
├── inference/          # ONNX inference
│   ├── session.go      # Session wrapper
│   └── pool.go         # Session pool
├── internal/proto/     # Generated protobuf
├── cmd/sat-cli/        # CLI tool
├── testdata/           # Test fixtures
└── docs/               # Documentation
```

## Documentation

- Update README.md for user-facing changes
- Update package doc comments for API changes
- Add examples for new features
- Keep ARCHITECTURE.md current for structural changes

## Questions

Open an issue for questions or discussion before starting significant work.
