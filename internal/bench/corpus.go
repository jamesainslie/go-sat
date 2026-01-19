// Package bench provides benchmarking utilities for sentence boundary detection.
package bench

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
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

// Sentence represents a parsed sentence with character offsets.
type Sentence struct {
	Text  string
	Start int
	End   int
}

// Common abbreviations that shouldn't end sentences
var abbreviations = regexp.MustCompile(`(?i)\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g|U\.S|U\.K)\.$`)

// ParseSentences splits text into sentences at sentence-ending punctuation.
// Handles common abbreviations to avoid false splits.
func ParseSentences(text string) []Sentence {
	if text == "" {
		return nil
	}

	var sentences []Sentence
	start := 0

	for i := 0; i < len(text); i++ {
		ch := text[i]
		if ch == '.' || ch == '?' || ch == '!' {
			// Check if this is end of text or followed by space/newline
			isEnd := i == len(text)-1 || text[i+1] == ' ' || text[i+1] == '\n'
			if !isEnd {
				continue
			}

			// Check for abbreviation
			candidate := text[start : i+1]
			if ch == '.' && abbreviations.MatchString(candidate) {
				continue
			}

			end := i + 1
			sentences = append(sentences, Sentence{
				Text:  strings.TrimSpace(text[start:end]),
				Start: start,
				End:   end,
			})

			// Skip whitespace to find next sentence start
			for i+1 < len(text) && (text[i+1] == ' ' || text[i+1] == '\n') {
				i++
			}
			start = i + 1
		}
	}

	// Handle remaining text without terminal punctuation
	if start < len(text) {
		remaining := strings.TrimSpace(text[start:])
		if remaining != "" {
			sentences = append(sentences, Sentence{
				Text:  remaining,
				Start: start,
				End:   len(text),
			})
		}
	}

	return sentences
}

// Talk represents a loaded transcript with parsed sentences.
type Talk struct {
	ID        string     // filename without extension
	Source    string     // TED URL
	Speaker   string
	Title     string
	RawText   string     // body text
	Sentences []Sentence
}

// LoadTalk loads and parses a transcript file.
func LoadTalk(path string) (*Talk, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	header, body, err := ParseHeader(string(data))
	if err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}

	base := filepath.Base(path)
	id := strings.TrimSuffix(base, filepath.Ext(base))

	return &Talk{
		ID:        id,
		Source:    header.Source,
		Speaker:   header.Speaker,
		Title:     header.Title,
		RawText:   body,
		Sentences: ParseSentences(body),
	}, nil
}

// LoadCorpus loads all .txt transcript files from a directory.
func LoadCorpus(dir string) ([]*Talk, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read dir: %w", err)
	}

	var talks []*Talk
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if filepath.Ext(entry.Name()) != ".txt" {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		talk, err := LoadTalk(path)
		if err != nil {
			return nil, fmt.Errorf("loading %s: %w", entry.Name(), err)
		}
		talks = append(talks, talk)
	}

	return talks, nil
}
