package inference

import (
	"context"
	"errors"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewPool_InvalidSize(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	// Size <= 0 should default to 1
	pool, err := NewPool(modelPath, 0)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewPool failed: %v", err)
	}
	defer func() { _ = pool.Close() }()

	if pool.Size() != 1 {
		t.Errorf("expected size 1 for invalid input, got %d", pool.Size())
	}
}

func TestNewPool_NegativeSize(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	// Negative size should default to 1
	pool, err := NewPool(modelPath, -5)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewPool failed: %v", err)
	}
	defer func() { _ = pool.Close() }()

	if pool.Size() != 1 {
		t.Errorf("expected size 1 for negative input, got %d", pool.Size())
	}
}

func TestNewPool_ModelNotFound(t *testing.T) {
	_, err := NewPool("../testdata/nonexistent.onnx", 2)
	if err == nil {
		t.Error("expected error for non-existent model file")
	}
}

func TestPool_AcquireRelease(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	pool, err := NewPool(modelPath, 2)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewPool failed: %v", err)
	}
	defer func() { _ = pool.Close() }()

	ctx := context.Background()

	// Acquire first session
	s1, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire 1 failed: %v", err)
	}

	// Acquire second session
	s2, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire 2 failed: %v", err)
	}

	// Third acquire should block - test with timeout
	ctx3, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()

	_, err = pool.Acquire(ctx3)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected DeadlineExceeded, got %v", err)
	}

	// Release one and acquire again should work
	pool.Release(s1)

	s3, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire 3 failed: %v", err)
	}

	pool.Release(s2)
	pool.Release(s3)
}

func TestPool_ReleaseNil(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	pool, err := NewPool(modelPath, 1)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewPool failed: %v", err)
	}
	defer func() { _ = pool.Close() }()

	// Should not panic when releasing nil
	pool.Release(nil)
}

func TestPool_Close_Idempotent(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	pool, err := NewPool(modelPath, 2)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewPool failed: %v", err)
	}

	// First close should succeed
	if err := pool.Close(); err != nil {
		t.Errorf("first Close failed: %v", err)
	}

	// Second close should also succeed (idempotent)
	if err := pool.Close(); err != nil {
		t.Errorf("second Close failed: %v", err)
	}
}

func TestPool_ReleaseAfterClose(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	pool, err := NewPool(modelPath, 1)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewPool failed: %v", err)
	}

	ctx := context.Background()
	session, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}

	// Close pool while session is out
	if err := pool.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Release should close the session instead of returning it to pool
	// This should not panic
	pool.Release(session)
}

func TestPool_AcquireContextCancellation(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	pool, err := NewPool(modelPath, 1)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewPool failed: %v", err)
	}
	defer func() { _ = pool.Close() }()

	ctx := context.Background()

	// Acquire the only session
	s1, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire 1 failed: %v", err)
	}
	defer pool.Release(s1)

	// Create a pre-cancelled context
	cancelledCtx, cancel := context.WithCancel(ctx)
	cancel()

	_, err = pool.Acquire(cancelledCtx)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestPool_ConcurrentAccess(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	poolSize := 3
	pool, err := NewPool(modelPath, poolSize)
	if err != nil {
		if isORTUnavailableError(err) {
			t.Skipf("Skipping: ONNX runtime not available: %v", err)
		}
		t.Fatalf("NewPool failed: %v", err)
	}
	defer func() { _ = pool.Close() }()

	ctx := context.Background()
	numGoroutines := 10
	numIterations := 5

	var wg sync.WaitGroup
	var successCount int64
	var errCount int64

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				// Use short timeout to avoid blocking forever
				acquireCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
				session, err := pool.Acquire(acquireCtx)
				cancel()

				if err != nil {
					atomic.AddInt64(&errCount, 1)
					continue
				}

				// Simulate some work
				time.Sleep(time.Millisecond)

				pool.Release(session)
				atomic.AddInt64(&successCount, 1)
			}
		}()
	}

	wg.Wait()

	// We should have had at least some successes
	if successCount == 0 {
		t.Error("expected at least some successful acquire/release cycles")
	}

	t.Logf("Concurrent test completed: %d successes, %d timeouts", successCount, errCount)
}

func TestPool_Size(t *testing.T) {
	modelPath := "../testdata/model_optimized.onnx"

	// Skip if model file doesn't exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Skipping: model not available at %s", modelPath)
	}

	sizes := []int{1, 2, 5}
	for _, size := range sizes {
		pool, err := NewPool(modelPath, size)
		if err != nil {
			if isORTUnavailableError(err) {
				t.Skipf("Skipping: ONNX runtime not available: %v", err)
			}
			t.Fatalf("NewPool failed for size %d: %v", size, err)
		}

		if got := pool.Size(); got != size {
			t.Errorf("Size() = %d, want %d", got, size)
		}

		_ = pool.Close()
	}
}
