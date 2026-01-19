// Package bench provides benchmarking utilities for sentence boundary detection.
package bench

import (
	"bufio"
	"errors"
	"fmt"
	"strings"
)

// Header contains metadata parsed from transcript file header.
type Header struct {
	Source  string
	Speaker string
	Title   string
}

// ParseHeader extracts metadata from transcript header comments.
// Returns the header, remaining text after header, and any error.
func ParseHeader(text string) (Header, string, error) {
	var h Header
	scanner := bufio.NewScanner(strings.NewReader(text))
	var bodyStart int
	var lineEnd int

	for scanner.Scan() {
		line := scanner.Text()
		lineEnd += len(line) + 1 // +1 for newline

		if !strings.HasPrefix(line, "#") {
			if strings.TrimSpace(line) == "" {
				continue
			}
			bodyStart = lineEnd - len(line) - 1
			break
		}

		line = strings.TrimPrefix(line, "# ")
		if value, ok := strings.CutPrefix(line, "Source:"); ok {
			h.Source = strings.TrimSpace(value)
		} else if value, ok := strings.CutPrefix(line, "Speaker:"); ok {
			h.Speaker = strings.TrimSpace(value)
		} else if value, ok := strings.CutPrefix(line, "Title:"); ok {
			h.Title = strings.TrimSpace(value)
		}
	}

	if err := scanner.Err(); err != nil {
		return Header{}, "", fmt.Errorf("scan header: %w", err)
	}

	if h.Source == "" {
		return Header{}, "", errors.New("missing Source in header")
	}

	body := text[bodyStart:]
	body = strings.TrimSpace(body)

	return h, body, nil
}
