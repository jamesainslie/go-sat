package tokenizer

import (
	"testing"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Hello", "▁Hello"},
		{"Hello world", "▁Hello▁world"},
		{"  spaces  ", "▁spaces"},
		{"", ""},
	}

	for _, tc := range tests {
		got := normalize(tc.input)
		if got != tc.expected {
			t.Errorf("normalize(%q) = %q, want %q", tc.input, got, tc.expected)
		}
	}
}
