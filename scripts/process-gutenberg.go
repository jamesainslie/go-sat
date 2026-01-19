//go:build ignore

// Process raw Project Gutenberg downloads into corpus format.
// Usage: go run ./scripts/process-gutenberg.go
package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Book metadata
var books = map[string]struct {
	Title  string
	Author string
	Year   string
}{
	"pride_and_prejudice": {"Pride and Prejudice", "Jane Austen", "1813"},
	"moby_dick":           {"Moby Dick", "Herman Melville", "1851"},
	"great_expectations":  {"Great Expectations", "Charles Dickens", "1861"},
	"origin_of_species":   {"On the Origin of Species", "Charles Darwin", "1859"},
	"tom_sawyer":          {"The Adventures of Tom Sawyer", "Mark Twain", "1876"},
	"jane_eyre":           {"Jane Eyre", "Charlotte Brontë", "1847"},
}

func main() {
	inDir := "testdata/gutenberg"

	files, err := filepath.Glob(filepath.Join(inDir, "*_raw.txt"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error finding files: %v\n", err)
		os.Exit(1)
	}

	if len(files) == 0 {
		fmt.Println("No raw files found. Run ./scripts/fetch-gutenberg.sh first.")
		os.Exit(1)
	}

	for _, rawFile := range files {
		baseName := strings.TrimSuffix(filepath.Base(rawFile), "_raw.txt")
		outFile := filepath.Join(inDir, baseName+".txt")

		meta, ok := books[baseName]
		if !ok {
			fmt.Printf("Skipping unknown book: %s\n", baseName)
			continue
		}

		fmt.Printf("Processing %s...\n", baseName)
		if err := processBook(rawFile, outFile, meta.Title, meta.Author, meta.Year); err != nil {
			fmt.Fprintf(os.Stderr, "Error processing %s: %v\n", baseName, err)
			continue
		}
		fmt.Printf("  -> %s\n", outFile)
	}

	fmt.Println("\nDone! Corpus files created in testdata/gutenberg/")
}

func processBook(inPath, outPath, title, author, year string) error {
	content, err := os.ReadFile(inPath)
	if err != nil {
		return fmt.Errorf("reading file: %w", err)
	}

	text := string(content)

	// Find the start marker
	startPatterns := []string{
		"*** START OF THE PROJECT GUTENBERG EBOOK",
		"*** START OF THIS PROJECT GUTENBERG EBOOK",
		"*END*THE SMALL PRINT",
	}

	startIdx := -1
	for _, pattern := range startPatterns {
		if idx := strings.Index(text, pattern); idx != -1 {
			// Find the end of this line
			endOfLine := strings.Index(text[idx:], "\n")
			if endOfLine != -1 {
				startIdx = idx + endOfLine + 1
			}
			break
		}
	}

	if startIdx == -1 {
		// Try to find first real content (skip initial blank lines and boilerplate)
		startIdx = 0
	}

	// Find the end marker
	endPatterns := []string{
		"*** END OF THE PROJECT GUTENBERG EBOOK",
		"*** END OF THIS PROJECT GUTENBERG EBOOK",
		"End of Project Gutenberg",
		"End of the Project Gutenberg",
	}

	endIdx := len(text)
	for _, pattern := range endPatterns {
		if idx := strings.Index(text, pattern); idx != -1 {
			endIdx = idx
			break
		}
	}

	// Extract the body
	body := text[startIdx:endIdx]

	// Clean up the body
	body = cleanBody(body)

	// Limit to first ~50KB for reasonable benchmark size
	if len(body) > 50000 {
		// Find a sentence boundary near the limit
		cutoff := 50000
		for i := cutoff; i < len(body) && i < cutoff+1000; i++ {
			if body[i] == '.' || body[i] == '!' || body[i] == '?' {
				if i+1 < len(body) && (body[i+1] == ' ' || body[i+1] == '\n') {
					body = body[:i+1]
					break
				}
			}
		}
	}

	// Create output with header
	out, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("creating output: %w", err)
	}
	defer out.Close()

	w := bufio.NewWriter(out)
	fmt.Fprintf(w, "# Source: https://www.gutenberg.org/\n")
	fmt.Fprintf(w, "# Author: %s\n", author)
	fmt.Fprintf(w, "# Title: %s (%s)\n", title, year)
	fmt.Fprintf(w, "\n")
	w.WriteString(body)
	w.WriteString("\n")

	return w.Flush()
}

func cleanBody(text string) string {
	// Normalize line endings
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	// Find where actual content starts using regex for chapter markers
	// Matches: "Chapter I", "CHAPTER 1", "CHAPTER I.", "Chapter 1.", etc.
	// Also handles: "Chapter I.]", "CHAPTER 1. Loomings", "CHAPTER I. VARIATION..."
	chapterRe := regexp.MustCompile(`(?m)^(Chapter|CHAPTER)\s+([IVX]+|[0-9]+)[\.\]\s]`)

	if loc := chapterRe.FindStringIndex(text); loc != nil && loc[0] < 50000 {
		text = text[loc[0]:]
	}

	// Remove illustration markers and other formatting artifacts
	illustrationRe := regexp.MustCompile(`\[Illustration[^\]]*\]`)
	text = illustrationRe.ReplaceAllString(text, "")

	// Remove excessive blank lines (more than 2 consecutive)
	multiBlank := regexp.MustCompile(`\n{3,}`)
	text = multiBlank.ReplaceAllString(text, "\n\n")

	// Trim leading/trailing whitespace
	text = strings.TrimSpace(text)

	// Join lines that are part of the same paragraph
	lines := strings.Split(text, "\n")
	var result []string
	var paragraph strings.Builder

	for _, line := range lines {
		line = strings.TrimRight(line, " \t")

		// Skip lines that look like chapter headers or page numbers in the middle
		if isChapterHeader(line) && paragraph.Len() > 0 {
			// Save current paragraph and add chapter header
			result = append(result, paragraph.String())
			paragraph.Reset()
			result = append(result, line)
			continue
		}

		if line == "" {
			// End of paragraph
			if paragraph.Len() > 0 {
				result = append(result, paragraph.String())
				paragraph.Reset()
			}
			continue
		}

		if paragraph.Len() > 0 {
			paragraph.WriteString(" ")
		}
		paragraph.WriteString(line)
	}

	// Don't forget the last paragraph
	if paragraph.Len() > 0 {
		result = append(result, paragraph.String())
	}

	return strings.Join(result, "\n\n")
}

func isChapterHeader(line string) bool {
	line = strings.TrimSpace(line)
	if strings.HasPrefix(line, "CHAPTER ") || strings.HasPrefix(line, "Chapter ") {
		return true
	}
	// Roman numeral chapters
	if matched, _ := regexp.MatchString(`^[IVXLC]+\.?$`, line); matched {
		return true
	}
	return false
}
