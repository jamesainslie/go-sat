package bench

import (
	"testing"
)

func TestParseHeader(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		want     Header
		wantBody string
		wantErr  bool
	}{
		{
			name: "valid header",
			input: `# Source: https://example.com/talk
# Speaker: John Doe
# Title: My Talk

Hello world.`,
			want: Header{
				Source:  "https://example.com/talk",
				Speaker: "John Doe",
				Title:   "My Talk",
			},
			wantBody: "Hello world.",
		},
		{
			name: "missing source",
			input: `# Speaker: John Doe
# Title: My Talk

Hello.`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, body, err := ParseHeader(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseHeader() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			if got != tt.want {
				t.Errorf("ParseHeader() header = %+v, want %+v", got, tt.want)
			}
			if body != tt.wantBody {
				t.Errorf("ParseHeader() body = %q, want %q", body, tt.wantBody)
			}
		})
	}
}

func TestParseSentences(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []Sentence
	}{
		{
			name:  "simple sentences",
			input: "Hello world. How are you?",
			want: []Sentence{
				{Text: "Hello world.", Start: 0, End: 12},
				{Text: "How are you?", Start: 13, End: 25},
			},
		},
		{
			name:  "exclamation",
			input: "Wow! That's great.",
			want: []Sentence{
				{Text: "Wow!", Start: 0, End: 4},
				{Text: "That's great.", Start: 5, End: 18},
			},
		},
		{
			name:  "abbreviation Mr.",
			input: "Mr. Smith went home. He was tired.",
			want: []Sentence{
				{Text: "Mr. Smith went home.", Start: 0, End: 20},
				{Text: "He was tired.", Start: 21, End: 34},
			},
		},
		{
			name:  "abbreviation Dr.",
			input: "Dr. Jones called. She left a message.",
			want: []Sentence{
				{Text: "Dr. Jones called.", Start: 0, End: 17},
				{Text: "She left a message.", Start: 18, End: 37},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseSentences(tt.input)
			if len(got) != len(tt.want) {
				t.Errorf("ParseSentences() got %d sentences, want %d", len(got), len(tt.want))
				for i, s := range got {
					t.Logf("  got[%d]: %+v", i, s)
				}
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("sentence[%d] = %+v, want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}
}
