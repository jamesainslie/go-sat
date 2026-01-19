//go:build ignore

// Process UD English Web Treebank CoNLL-U files into benchmark corpus format.
// Creates JSON files with text and gold-standard sentence boundary positions.
// Usage: go run ./scripts/process-ud-ewt.go
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Corpus represents a processed corpus with gold standard annotations.
type Corpus struct {
	Name       string `json:"name"`
	Source     string `json:"source"`
	Text       string `json:"text"`
	Sentences  int    `json:"sentences"`
	Boundaries []int  `json:"boundaries"` // Character offsets where sentences end
}

func main() {
	inDir := "testdata/ud-ewt"
	outDir := "testdata/ud-ewt"

	splits := []string{"train", "dev", "test"}

	for _, split := range splits {
		inFile := filepath.Join(inDir, fmt.Sprintf("en_ewt-ud-%s.conllu", split))
		outFile := filepath.Join(outDir, fmt.Sprintf("%s.json", split))

		fmt.Printf("Processing %s...\n", split)
		corpus, err := processCoNLLU(inFile, split)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error processing %s: %v\n", inFile, err)
			continue
		}

		if err := writeCorpus(outFile, corpus); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing %s: %v\n", outFile, err)
			continue
		}

		fmt.Printf("  -> %s (%d sentences, %d chars)\n", outFile, corpus.Sentences, len(corpus.Text))
	}

	// Also create a combined corpus from all splits
	fmt.Println("Creating combined corpus...")
	if err := createCombined(inDir, outDir); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating combined corpus: %v\n", err)
	}

	fmt.Println("\nDone! Corpus files created in testdata/ud-ewt/")
}

func processCoNLLU(path, name string) (*Corpus, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}
	defer file.Close()

	var (
		sentences  []string
		currentTxt string
	)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()

		// Metadata line with sentence text
		if strings.HasPrefix(line, "# text = ") {
			currentTxt = strings.TrimPrefix(line, "# text = ")
			continue
		}

		// Blank line = end of sentence
		if line == "" && currentTxt != "" {
			sentences = append(sentences, currentTxt)
			currentTxt = ""
			continue
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanning file: %w", err)
	}

	// Don't forget last sentence if no trailing blank
	if currentTxt != "" {
		sentences = append(sentences, currentTxt)
	}

	// Build combined text and calculate boundaries
	var text strings.Builder
	var boundaries []int
	offset := 0

	for i, sent := range sentences {
		if i > 0 {
			text.WriteString(" ")
			offset++
		}
		text.WriteString(sent)
		offset += len(sent)
		boundaries = append(boundaries, offset) // Position after last char of sentence
	}

	return &Corpus{
		Name:       fmt.Sprintf("UD-EWT-%s", name),
		Source:     "https://github.com/UniversalDependencies/UD_English-EWT",
		Text:       text.String(),
		Sentences:  len(sentences),
		Boundaries: boundaries,
	}, nil
}

func writeCorpus(path string, corpus *Corpus) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(corpus)
}

func createCombined(inDir, outDir string) error {
	var allSentences []string

	splits := []string{"train", "dev", "test"}

	for _, split := range splits {
		path := filepath.Join(inDir, fmt.Sprintf("%s.json", split))
		file, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("opening %s: %w", path, err)
		}

		var corpus Corpus
		if err := json.NewDecoder(file).Decode(&corpus); err != nil {
			file.Close()
			return fmt.Errorf("decoding %s: %w", path, err)
		}
		file.Close()

		// Re-extract sentences from text using boundaries
		text := corpus.Text
		prev := 0
		for _, boundary := range corpus.Boundaries {
			if boundary <= len(text) {
				sent := strings.TrimSpace(text[prev:boundary])
				if sent != "" {
					allSentences = append(allSentences, sent)
				}
				prev = boundary
			}
		}
	}

	// Rebuild combined text and boundaries
	var text strings.Builder
	var boundaries []int
	offset := 0

	for i, sent := range allSentences {
		if i > 0 {
			text.WriteString(" ")
			offset++
		}
		text.WriteString(sent)
		offset += len(sent)
		boundaries = append(boundaries, offset)
	}

	combined := &Corpus{
		Name:       "UD-EWT-combined",
		Source:     "https://github.com/UniversalDependencies/UD_English-EWT",
		Text:       text.String(),
		Sentences:  len(allSentences),
		Boundaries: boundaries,
	}

	outPath := filepath.Join(outDir, "combined.json")
	if err := writeCorpus(outPath, combined); err != nil {
		return err
	}

	fmt.Printf("  -> %s (%d sentences, %d chars)\n", outPath, combined.Sentences, len(combined.Text))
	return nil
}
