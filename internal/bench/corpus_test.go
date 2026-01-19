package bench

import (
	"os"
	"path/filepath"
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

func TestLoadTalk(t *testing.T) {
	// Create temp file
	dir := t.TempDir()
	path := filepath.Join(dir, "test_talk.txt")
	content := `# Source: https://example.com
# Speaker: Test Speaker
# Title: Test Title

Hello world. How are you?`

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	talk, err := LoadTalk(path)
	if err != nil {
		t.Fatalf("LoadTalk() error = %v", err)
	}

	if talk.ID != "test_talk" {
		t.Errorf("ID = %q, want %q", talk.ID, "test_talk")
	}
	if talk.Speaker != "Test Speaker" {
		t.Errorf("Speaker = %q, want %q", talk.Speaker, "Test Speaker")
	}
	if len(talk.Sentences) != 2 {
		t.Errorf("got %d sentences, want 2", len(talk.Sentences))
	}
}

func TestLoadCorpus(t *testing.T) {
	dir := t.TempDir()

	// Create two test files
	for _, name := range []string{"talk1.txt", "talk2.txt"} {
		content := `# Source: https://example.com
# Speaker: Speaker
# Title: Title

Hello.`
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}

	// Create a non-txt file that should be ignored
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# Readme"), 0644); err != nil {
		t.Fatal(err)
	}

	talks, err := LoadCorpus(dir)
	if err != nil {
		t.Fatalf("LoadCorpus() error = %v", err)
	}

	if len(talks) != 2 {
		t.Errorf("got %d talks, want 2", len(talks))
	}
}
