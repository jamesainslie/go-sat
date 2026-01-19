package tokenizer

import (
	"testing"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"simple word", "Hello", "▁Hello"},
		{"two words", "Hello world", "▁Hello▁world"},
		{"extra spaces", "  spaces  ", "▁spaces"},
		{"empty string", "", ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := normalize(tc.input)
			if got != tc.expected {
				t.Errorf("normalize(%q) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}
}
