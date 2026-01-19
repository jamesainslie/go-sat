package tokenizer

import (
	"strings"
	"unicode"
)

const sentencePieceSpace = '▁' // U+2581 LOWER ONE EIGHTH BLOCK

// normalize prepares text for tokenization following XLM-RoBERTa conventions.
// - Adds dummy prefix (space at start)
// - Replaces spaces with ▁
// - Normalizes whitespace (collapses runs, trims trailing)
func normalize(text string) string {
	if text == "" {
		return ""
	}

	// Normalize whitespace: collapse runs, trim trailing
	var builder strings.Builder
	needSpace := true // start true to add dummy prefix before first non-space

	for _, r := range text {
		if unicode.IsSpace(r) {
			// Mark that we need a space before the next non-space char
			// (only if we've already written something)
			if builder.Len() > 0 {
				needSpace = true
			}
		} else {
			// Write pending space separator before this character
			if needSpace {
				builder.WriteRune(sentencePieceSpace)
				needSpace = false
			}
			builder.WriteRune(r)
		}
	}

	return builder.String()
}
