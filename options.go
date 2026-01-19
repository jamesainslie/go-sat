package sat

import (
	"log/slog"
	"runtime"
)

// Option configures a Segmenter.
type Option func(*config)

type config struct {
	threshold float32
	poolSize  int
	logger    *slog.Logger
}

func defaultConfig() config {
	return config{
		threshold: 0.025,
		poolSize:  runtime.NumCPU(),
		logger:    slog.Default(),
	}
}

// WithThreshold sets the boundary detection threshold (default: 0.025).
func WithThreshold(t float32) Option {
	return func(c *config) {
		c.threshold = t
	}
}

// WithPoolSize sets the ONNX session pool size (default: runtime.NumCPU()).
func WithPoolSize(n int) Option {
	return func(c *config) {
		if n > 0 {
			c.poolSize = n
		}
	}
}

// WithLogger sets the logger (default: slog.Default()).
func WithLogger(l *slog.Logger) Option {
	return func(c *config) {
		if l != nil {
			c.logger = l
		}
	}
}
