package inference

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// Pool manages a pool of ONNX sessions for concurrent inference.
type Pool struct {
	sessions  chan *Session
	modelPath string
	size      int
	mu        sync.Mutex
	closed    bool
}

// NewPool creates a pool of n ONNX sessions.
func NewPool(modelPath string, size int) (*Pool, error) {
	if size <= 0 {
		size = 1
	}

	pool := &Pool{
		sessions:  make(chan *Session, size),
		modelPath: modelPath,
		size:      size,
	}

	// Pre-create all sessions
	for i := 0; i < size; i++ {
		session, err := NewSession(modelPath)
		if err != nil {
			// Clean up already created sessions
			_ = pool.Close() // Best-effort cleanup; original error takes precedence
			return nil, fmt.Errorf("creating session %d: %w", i, err)
		}
		pool.sessions <- session
	}

	return pool, nil
}

// Acquire gets a session from the pool, blocking if none available.
// Respects context cancellation. Returns error if pool is closed.
func (p *Pool) Acquire(ctx context.Context) (*Session, error) {
	select {
	case session, ok := <-p.sessions:
		if !ok {
			return nil, ErrPoolClosed
		}
		return session, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Release returns a session to the pool.
func (p *Pool) Release(s *Session) {
	if s == nil {
		return
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		_ = s.Close() // Pool closed; clean up session
		return
	}
	p.mu.Unlock()

	select {
	case p.sessions <- s:
	default:
		_ = s.Close() // Pool full; clean up excess session
	}
}

// Close closes all sessions in the pool.
func (p *Pool) Close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	p.mu.Unlock()

	close(p.sessions)

	var errs []error
	for session := range p.sessions {
		if err := session.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

// Size returns the pool size.
func (p *Pool) Size() int {
	return p.size
}
