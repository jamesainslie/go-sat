//go:build stave

package main

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/yaklabco/stave/pkg/sh"
	"github.com/yaklabco/stave/pkg/st"
	"github.com/yaklabco/stave/pkg/target"
)

// Default target when running `stave` with no arguments.
var Default = All

// Aliases for common targets.
var Aliases = map[string]interface{}{
	"b": Build,
	"t": Test,
	"l": Lint,
	"c": Clean,
}

// All runs the complete build pipeline: lint, test, and build.
func All() error {
	st.Deps(Init)
	st.Deps(Lint, Test)
	st.Deps(Build)
	return nil
}

// Init ensures the module dependencies are up to date.
func Init() error {
	return sh.Run("go", "mod", "tidy")
}

// Build compiles both sat-cli and sat-bench binaries.
func Build() error {
	st.Deps(Init)
	st.Deps(Build_CLI, Build_Bench)
	return nil
}

// Build_CLI compiles the sat-cli binary with version information.
func Build_CLI() error {
	st.Deps(Init)

	// Check if rebuild is needed
	rebuild, err := target.Glob("bin/sat-cli", "**/*.go", "go.mod", "go.sum")
	if err != nil {
		return fmt.Errorf("checking rebuild: %w", err)
	}
	if !rebuild {
		if st.Verbose() {
			fmt.Println("sat-cli is up to date")
		}
		return nil
	}

	ldflags := buildLdflags()
	return sh.RunV("go", "build", "-ldflags", ldflags, "-o", "bin/sat-cli", "./cmd/sat-cli")
}

// Build_Bench compiles the sat-bench binary with version information.
func Build_Bench() error {
	st.Deps(Init)

	// Check if rebuild is needed
	rebuild, err := target.Glob("bin/sat-bench", "**/*.go", "go.mod", "go.sum")
	if err != nil {
		return fmt.Errorf("checking rebuild: %w", err)
	}
	if !rebuild {
		if st.Verbose() {
			fmt.Println("sat-bench is up to date")
		}
		return nil
	}

	ldflags := buildLdflags()
	return sh.RunV("go", "build", "-ldflags", ldflags, "-o", "bin/sat-bench", "./cmd/sat-bench")
}

// buildLdflags returns ldflags for version injection.
func buildLdflags() string {
	version, _ := sh.Output("git", "describe", "--tags", "--always", "--dirty")
	commit, _ := sh.Output("git", "rev-parse", "--short", "HEAD")
	date := time.Now().Format(time.RFC3339)

	return fmt.Sprintf(
		"-X main.version=%s -X main.commit=%s -X main.date=%s",
		strings.TrimSpace(version),
		strings.TrimSpace(commit),
		date,
	)
}

// Test runs all tests with race detection and coverage.
func Test() error {
	st.Deps(Init)
	return sh.RunV("go", "test", "-race", "-cover", "./...")
}

// TestShort runs tests in short mode (skips long-running tests).
func TestShort() error {
	st.Deps(Init)
	return sh.RunV("go", "test", "-short", "-race", "./...")
}

// TestVerbose runs tests with verbose output.
func TestVerbose() error {
	st.Deps(Init)
	return sh.RunV("go", "test", "-race", "-cover", "-v", "./...")
}

// Lint runs golangci-lint on the codebase.
func Lint() error {
	return sh.RunV("golangci-lint", "run", "./...")
}

// LintFix runs golangci-lint with auto-fix enabled.
func LintFix() error {
	return sh.RunV("golangci-lint", "run", "--fix", "./...")
}

// Fmt formats all Go code using gofmt and goimports.
func Fmt() error {
	if err := sh.Run("gofmt", "-w", "."); err != nil {
		return fmt.Errorf("gofmt: %w", err)
	}
	if err := sh.Run("goimports", "-w", "."); err != nil {
		return fmt.Errorf("goimports: %w", err)
	}
	return nil
}

// Vet runs go vet on all packages.
func Vet() error {
	return sh.RunV("go", "vet", "./...")
}

// Clean removes build artifacts.
func Clean() error {
	artifacts := []string{
		"bin/",
		"sat-bench",
		"sat-cli",
	}
	for _, a := range artifacts {
		if err := sh.Rm(a); err != nil {
			return fmt.Errorf("removing %s: %w", a, err)
		}
	}
	return nil
}

// Install builds and installs the binaries to GOBIN.
func Install() error {
	st.Deps(Build)

	gocmd := st.GoCmd()
	bin, err := sh.Output(gocmd, "env", "GOBIN")
	if err != nil {
		return fmt.Errorf("determining GOBIN: %w", err)
	}
	if bin == "" {
		gopath, err := sh.Output(gocmd, "env", "GOPATH")
		if err != nil {
			return fmt.Errorf("determining GOPATH: %w", err)
		}
		bin = gopath + "/bin"
	}

	binaries := []string{"sat-cli", "sat-bench"}
	for _, name := range binaries {
		src := "bin/" + name
		dst := bin + "/" + name
		if runtime.GOOS == "windows" {
			dst += ".exe"
		}
		if err := sh.Copy(dst, src); err != nil {
			return fmt.Errorf("installing %s: %w", name, err)
		}
		if st.Verbose() {
			fmt.Printf("Installed %s to %s\n", name, dst)
		}
	}
	return nil
}

// Proto namespace for protobuf-related targets.
type Proto st.Namespace

// Generate regenerates the protobuf Go code from .proto files.
func (Proto) Generate() error {
	protoFile := "internal/proto/sentencepiece_model.proto"
	outDir := "internal/proto"

	// Check if the .proto file exists
	if _, err := os.Stat(protoFile); os.IsNotExist(err) {
		return fmt.Errorf("proto file not found: %s", protoFile)
	}

	return sh.RunV("protoc",
		"--go_out="+outDir,
		"--go_opt=paths=source_relative",
		protoFile,
	)
}

// Bench namespace for benchmark-related targets.
type Bench st.Namespace

// Run runs the benchmark tool against the test corpus.
// Requires model.onnx and testdata/sentencepiece.bpe.model to exist.
func (Bench) Run() error {
	st.Deps(Build_Bench)

	modelPath := os.Getenv("SAT_MODEL")
	if modelPath == "" {
		modelPath = "model.onnx"
	}
	tokenizerPath := os.Getenv("SAT_TOKENIZER")
	if tokenizerPath == "" {
		tokenizerPath = "testdata/sentencepiece.bpe.model"
	}

	return sh.RunV("./bin/sat-bench",
		"-model", modelPath,
		"-tokenizer", tokenizerPath,
		"-corpus", "testdata/ted",
	)
}

// Sweep runs a threshold sweep to find optimal parameters.
func (Bench) Sweep() error {
	st.Deps(Build_Bench)

	modelPath := os.Getenv("SAT_MODEL")
	if modelPath == "" {
		modelPath = "model.onnx"
	}
	tokenizerPath := os.Getenv("SAT_TOKENIZER")
	if tokenizerPath == "" {
		tokenizerPath = "testdata/sentencepiece.bpe.model"
	}

	return sh.RunV("./bin/sat-bench",
		"-model", modelPath,
		"-tokenizer", tokenizerPath,
		"-corpus", "testdata/ted",
		"-sweep",
	)
}

// CI runs the full CI pipeline (lint, test, build).
func CI() error {
	st.Deps(Init)
	st.SerialDeps(Lint, Test, Build)
	return nil
}

// Check runs quick validation (vet, lint, short tests).
func Check() error {
	st.Deps(Vet, Lint, TestShort)
	return nil
}

// Coverage generates a coverage report.
func Coverage() error {
	st.Deps(Init)
	if err := sh.RunV("go", "test", "-race", "-coverprofile=coverage.out", "./..."); err != nil {
		return err
	}
	return sh.RunV("go", "tool", "cover", "-html=coverage.out", "-o", "coverage.html")
}

// Tidy runs go mod tidy and verifies the go.sum is clean.
func Tidy() error {
	if err := sh.Run("go", "mod", "tidy"); err != nil {
		return err
	}
	// Verify no changes to go.sum (useful for CI)
	output, err := sh.Output("git", "diff", "--exit-code", "go.sum")
	if err != nil {
		if output != "" {
			return fmt.Errorf("go.sum is not clean:\n%s", output)
		}
	}
	return nil
}
