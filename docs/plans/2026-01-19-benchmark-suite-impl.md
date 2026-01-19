# Benchmark Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `sat-bench` CLI that measures sentence boundary detection accuracy against TED Talk transcripts with configurable precision/recall weighting and threshold sweep capability.

**Architecture:** Corpus loader parses TED transcripts into sentences, metrics module compares predicted vs ground-truth boundaries, sweep module iterates thresholds to find optimal. CLI ties it together with flag-based configuration.

**Tech Stack:** Go stdlib (flag, testing), go-sat library for segmentation

---

## Task 1: Create Test Fixtures Directory

**Files:**
- Create: `testdata/ted/README.md`

**Step 1: Create directory and README**

```bash
mkdir -p testdata/ted
```

Create `testdata/ted/README.md`:
```markdown
# TED Talk Transcripts

Test fixtures for sentence boundary detection benchmarking.

## Sources

All transcripts downloaded from official TED.com with attribution.

| File | Speaker | Talk | URL |
|------|---------|------|-----|
| ken_robinson.txt | Sir Ken Robinson | Do Schools Kill Creativity? | https://www.ted.com/talks/sir_ken_robinson_do_schools_kill_creativity/transcript |
| brene_brown.txt | Brené Brown | The Power of Vulnerability | https://www.ted.com/talks/brene_brown_the_power_of_vulnerability/transcript |
| simon_sinek.txt | Simon Sinek | How Great Leaders Inspire Action | https://www.ted.com/talks/simon_sinek_how_great_leaders_inspire_action/transcript |
| jill_taylor.txt | Jill Bolte Taylor | My Stroke of Insight | https://www.ted.com/talks/jill_bolte_taylor_my_stroke_of_insight/transcript |
| chimamanda_adichie.txt | Chimamanda Ngozi Adichie | The Danger of a Single Story | https://www.ted.com/talks/chimamanda_ngozi_adichie_the_danger_of_a_single_story/transcript |
| amy_cuddy.txt | Amy Cuddy | Your Body Language May Shape Who You Are | https://www.ted.com/talks/amy_cuddy_your_body_language_may_shape_who_you_are/transcript |

## Usage

These transcripts are used by `sat-bench` to evaluate sentence boundary detection accuracy.

## License

Transcripts are copyright TED Conferences LLC. Used for testing/research purposes only.
```

**Step 2: Commit**

```bash
git add testdata/ted/README.md
git commit -m "docs: add TED transcript fixtures README with attribution"
```

---

## Task 2: Add First TED Transcript (Ken Robinson)

**Files:**
- Create: `testdata/ted/ken_robinson.txt`

**Step 1: Create transcript file**

Create `testdata/ted/ken_robinson.txt` with header and opening paragraphs:
```
# Source: https://www.ted.com/talks/sir_ken_robinson_do_schools_kill_creativity/transcript
# Speaker: Sir Ken Robinson
# Title: Do Schools Kill Creativity?

Good morning. How are you? It's been great, hasn't it? I've been blown away by the whole thing. In fact, I'm leaving.

There have been three themes running through the conference, which are relevant to what I want to talk about. One is the extraordinary evidence of human creativity in all of the presentations that we've had and in all of the people here. Just the variety of it and the range of it. The second is that it's put us in a place where we have no idea what's going to happen, in terms of the future. No idea how this may play out.

I have an interest in education. Actually, what I find is everybody has an interest in education. Don't you? I find this very interesting. If you're at a dinner party, and you say you work in education -- actually, you're not often at dinner parties, frankly. If you work in education, you're not asked. And you're never asked back, curiously. That's strange to me. But if you are, and you say to somebody, you know, they say, "What do you do?" and you say you work in education, you can see the blood run from their face. They're like, "Oh my God," you know, "Why me?"

My contention is that creativity now is as important in education as literacy, and we should treat it with the same status.
```

NOTE: The full transcript should be downloaded from TED.com. For this plan, include at least 500 words to have meaningful test data.

**Step 2: Commit**

```bash
git add testdata/ted/ken_robinson.txt
git commit -m "test: add Ken Robinson TED transcript fixture"
```

---

## Task 3: Add Remaining TED Transcripts

**Files:**
- Create: `testdata/ted/brene_brown.txt`
- Create: `testdata/ted/simon_sinek.txt`
- Create: `testdata/ted/jill_taylor.txt`
- Create: `testdata/ted/chimamanda_adichie.txt`
- Create: `testdata/ted/amy_cuddy.txt`

**Step 1: Download and format each transcript**

Each file follows the same format:
```
# Source: [TED URL]
# Speaker: [Name]
# Title: [Talk Title]

[Transcript text with original punctuation preserved]
```

Download from the URLs in the README. Preserve original punctuation exactly - this is our ground truth.

**Step 2: Commit**

```bash
git add testdata/ted/*.txt
git commit -m "test: add remaining TED transcript fixtures"
```

---

## Task 4: Implement Corpus Loader - Types and Tests

**Files:**
- Create: `internal/bench/corpus.go`
- Create: `internal/bench/corpus_test.go`

**Step 1: Write failing test for ParseHeader**

Create `internal/bench/corpus_test.go`:
```go
package bench

import (
	"testing"
)

func TestParseHeader(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    Header
		wantErr bool
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
			got, _, err := ParseHeader(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseHeader() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("ParseHeader() = %+v, want %+v", got, tt.want)
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./internal/bench/... -v
```

Expected: FAIL (package doesn't exist)

**Step 3: Write minimal implementation**

Create `internal/bench/corpus.go`:
```go
// Package bench provides benchmarking utilities for sentence boundary detection.
package bench

import (
	"bufio"
	"errors"
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
		if strings.HasPrefix(line, "Source:") {
			h.Source = strings.TrimSpace(strings.TrimPrefix(line, "Source:"))
		} else if strings.HasPrefix(line, "Speaker:") {
			h.Speaker = strings.TrimSpace(strings.TrimPrefix(line, "Speaker:"))
		} else if strings.HasPrefix(line, "Title:") {
			h.Title = strings.TrimSpace(strings.TrimPrefix(line, "Title:"))
		}
	}

	if h.Source == "" {
		return Header{}, "", errors.New("missing Source in header")
	}

	body := text[bodyStart:]
	body = strings.TrimSpace(body)

	return h, body, nil
}
```

**Step 4: Run test to verify it passes**

```bash
go test ./internal/bench/... -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add internal/bench/corpus.go internal/bench/corpus_test.go
git commit -m "feat(bench): add corpus header parsing"
```

---

## Task 5: Implement Sentence Parsing

**Files:**
- Modify: `internal/bench/corpus.go`
- Modify: `internal/bench/corpus_test.go`

**Step 1: Write failing test for ParseSentences**

Add to `internal/bench/corpus_test.go`:
```go
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
```

**Step 2: Run test to verify it fails**

```bash
go test ./internal/bench/... -v -run TestParseSentences
```

Expected: FAIL (Sentence type and ParseSentences not defined)

**Step 3: Write minimal implementation**

Add to `internal/bench/corpus.go`:
```go
import (
	"regexp"
	// ... existing imports
)

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
```

**Step 4: Run test to verify it passes**

```bash
go test ./internal/bench/... -v -run TestParseSentences
```

Expected: PASS

**Step 5: Commit**

```bash
git add internal/bench/corpus.go internal/bench/corpus_test.go
git commit -m "feat(bench): add sentence parsing with abbreviation handling"
```

---

## Task 6: Implement Talk Loading

**Files:**
- Modify: `internal/bench/corpus.go`
- Modify: `internal/bench/corpus_test.go`

**Step 1: Write failing test for LoadTalk**

Add to `internal/bench/corpus_test.go`:
```go
import (
	"os"
	"path/filepath"
	// ... existing imports
)

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
```

**Step 2: Run test to verify it fails**

```bash
go test ./internal/bench/... -v -run TestLoadTalk
```

Expected: FAIL (Talk type and LoadTalk not defined)

**Step 3: Write minimal implementation**

Add to `internal/bench/corpus.go`:
```go
import (
	"os"
	"path/filepath"
	// ... existing imports
)

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
		return nil, err
	}

	header, body, err := ParseHeader(string(data))
	if err != nil {
		return nil, err
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
```

**Step 4: Run test to verify it passes**

```bash
go test ./internal/bench/... -v -run TestLoadTalk
```

Expected: PASS

**Step 5: Commit**

```bash
git add internal/bench/corpus.go internal/bench/corpus_test.go
git commit -m "feat(bench): add talk loading from transcript files"
```

---

## Task 7: Implement Corpus Loading

**Files:**
- Modify: `internal/bench/corpus.go`
- Modify: `internal/bench/corpus_test.go`

**Step 1: Write failing test for LoadCorpus**

Add to `internal/bench/corpus_test.go`:
```go
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
```

**Step 2: Run test to verify it fails**

```bash
go test ./internal/bench/... -v -run TestLoadCorpus
```

Expected: FAIL (LoadCorpus not defined)

**Step 3: Write minimal implementation**

Add to `internal/bench/corpus.go`:
```go
// LoadCorpus loads all .txt transcript files from a directory.
func LoadCorpus(dir string) ([]*Talk, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
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
```

Add `"fmt"` to imports.

**Step 4: Run test to verify it passes**

```bash
go test ./internal/bench/... -v -run TestLoadCorpus
```

Expected: PASS

**Step 5: Commit**

```bash
git add internal/bench/corpus.go internal/bench/corpus_test.go
git commit -m "feat(bench): add corpus loading from directory"
```

---

## Task 8: Implement Metrics Types and Calculation

**Files:**
- Create: `internal/bench/metrics.go`
- Create: `internal/bench/metrics_test.go`

**Step 1: Write failing test for Evaluate**

Create `internal/bench/metrics_test.go`:
```go
package bench

import "testing"

func TestEvaluate(t *testing.T) {
	tests := []struct {
		name      string
		predicted []int
		truth     []int
		tolerance int
		wantTP    int
		wantFP    int
		wantFN    int
	}{
		{
			name:      "perfect match",
			predicted: []int{10, 20, 30},
			truth:     []int{10, 20, 30},
			tolerance: 0,
			wantTP:    3,
			wantFP:    0,
			wantFN:    0,
		},
		{
			name:      "within tolerance",
			predicted: []int{11, 19, 31},
			truth:     []int{10, 20, 30},
			tolerance: 2,
			wantTP:    3,
			wantFP:    0,
			wantFN:    0,
		},
		{
			name:      "false positive",
			predicted: []int{10, 15, 20},
			truth:     []int{10, 20},
			tolerance: 0,
			wantTP:    2,
			wantFP:    1,
			wantFN:    0,
		},
		{
			name:      "false negative",
			predicted: []int{10},
			truth:     []int{10, 20},
			tolerance: 0,
			wantTP:    1,
			wantFP:    0,
			wantFN:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := Config{Tolerance: tt.tolerance}
			got := Evaluate(tt.predicted, tt.truth, cfg)

			if got.TruePositives != tt.wantTP {
				t.Errorf("TruePositives = %d, want %d", got.TruePositives, tt.wantTP)
			}
			if got.FalsePositives != tt.wantFP {
				t.Errorf("FalsePositives = %d, want %d", got.FalsePositives, tt.wantFP)
			}
			if got.FalseNegatives != tt.wantFN {
				t.Errorf("FalseNegatives = %d, want %d", got.FalseNegatives, tt.wantFN)
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./internal/bench/... -v -run TestEvaluate
```

Expected: FAIL (types not defined)

**Step 3: Write minimal implementation**

Create `internal/bench/metrics.go`:
```go
package bench

// Config holds evaluation parameters.
type Config struct {
	Threshold       float32
	Tolerance       int     // character match tolerance
	PrecisionWeight float64
	RecallWeight    float64
}

// DefaultConfig returns default evaluation configuration.
func DefaultConfig() Config {
	return Config{
		Threshold:       0.025,
		Tolerance:       3,
		PrecisionWeight: 1.0,
		RecallWeight:    1.0,
	}
}

// Metrics holds evaluation results.
type Metrics struct {
	TruePositives  int
	FalsePositives int
	FalseNegatives int
	Precision      float64
	Recall         float64
	F1             float64
	WeightedScore  float64
}

// Evaluate compares predicted boundaries against ground truth.
// Uses greedy left-to-right matching within tolerance.
func Evaluate(predicted, truth []int, cfg Config) Metrics {
	matched := make([]bool, len(truth))
	tp := 0

	for _, p := range predicted {
		for i, t := range truth {
			if matched[i] {
				continue
			}
			diff := p - t
			if diff < 0 {
				diff = -diff
			}
			if diff <= cfg.Tolerance {
				matched[i] = true
				tp++
				break
			}
		}
	}

	fp := len(predicted) - tp
	fn := len(truth) - tp

	m := Metrics{
		TruePositives:  tp,
		FalsePositives: fp,
		FalseNegatives: fn,
	}

	if tp+fp > 0 {
		m.Precision = float64(tp) / float64(tp+fp)
	}
	if tp+fn > 0 {
		m.Recall = float64(tp) / float64(tp+fn)
	}
	if m.Precision+m.Recall > 0 {
		m.F1 = 2 * m.Precision * m.Recall / (m.Precision + m.Recall)
	}

	wp := cfg.PrecisionWeight
	wr := cfg.RecallWeight
	if wp+wr > 0 {
		m.WeightedScore = (wp*m.Precision + wr*m.Recall) / (wp + wr)
	}

	return m
}
```

**Step 4: Run test to verify it passes**

```bash
go test ./internal/bench/... -v -run TestEvaluate
```

Expected: PASS

**Step 5: Commit**

```bash
git add internal/bench/metrics.go internal/bench/metrics_test.go
git commit -m "feat(bench): add metrics evaluation with tolerance matching"
```

---

## Task 9: Implement Talk Evaluation

**Files:**
- Modify: `internal/bench/metrics.go`
- Modify: `internal/bench/metrics_test.go`

**Step 1: Write failing test for EvaluateTalk**

Add to `internal/bench/metrics_test.go`:
```go
import (
	"context"
	"os"
	"testing"

	sat "github.com/jamesainslie/go-sat"
)

func TestEvaluateTalk(t *testing.T) {
	modelPath := os.Getenv("SAT_MODEL_PATH")
	tokenizerPath := os.Getenv("SAT_TOKENIZER_PATH")
	if modelPath == "" || tokenizerPath == "" {
		t.Skip("SAT_MODEL_PATH and SAT_TOKENIZER_PATH not set")
	}

	seg, err := sat.New(modelPath, tokenizerPath)
	if err != nil {
		t.Fatalf("failed to create segmenter: %v", err)
	}
	defer seg.Close()

	talk := &Talk{
		ID:      "test",
		RawText: "Hello world. How are you?",
		Sentences: []Sentence{
			{Text: "Hello world.", Start: 0, End: 12},
			{Text: "How are you?", Start: 13, End: 25},
		},
	}

	cfg := DefaultConfig()
	metrics, err := EvaluateTalk(context.Background(), seg, talk, cfg)
	if err != nil {
		t.Fatalf("EvaluateTalk() error = %v", err)
	}

	// Should get reasonable precision/recall on simple sentences
	if metrics.Precision < 0.5 {
		t.Errorf("Precision = %v, want >= 0.5", metrics.Precision)
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./internal/bench/... -v -run TestEvaluateTalk
```

Expected: FAIL (EvaluateTalk not defined) or SKIP if env vars not set

**Step 3: Write minimal implementation**

Add to `internal/bench/metrics.go`:
```go
import (
	"context"

	sat "github.com/jamesainslie/go-sat"
)

// EvaluateTalk runs segmentation on a talk and evaluates against ground truth.
func EvaluateTalk(ctx context.Context, seg *sat.Segmenter, talk *Talk, cfg Config) (Metrics, error) {
	// Get predicted boundaries
	sentences, err := seg.Segment(ctx, talk.RawText)
	if err != nil {
		return Metrics{}, err
	}

	// Convert to boundary positions (end of each sentence)
	var predicted []int
	pos := 0
	for _, s := range sentences {
		pos += len(s)
		predicted = append(predicted, pos)
	}

	// Get ground truth boundaries
	var truth []int
	for _, s := range talk.Sentences {
		truth = append(truth, s.End)
	}

	return Evaluate(predicted, truth, cfg), nil
}
```

**Step 4: Run test to verify it passes**

```bash
SAT_MODEL_PATH=/path/to/model.onnx SAT_TOKENIZER_PATH=/path/to/tokenizer.model go test ./internal/bench/... -v -run TestEvaluateTalk
```

Expected: PASS (or SKIP if env vars not set)

**Step 5: Commit**

```bash
git add internal/bench/metrics.go internal/bench/metrics_test.go
git commit -m "feat(bench): add talk evaluation against segmenter"
```

---

## Task 10: Implement Threshold Sweep

**Files:**
- Create: `internal/bench/sweep.go`
- Create: `internal/bench/sweep_test.go`

**Step 1: Write failing test for Sweep**

Create `internal/bench/sweep_test.go`:
```go
package bench

import (
	"testing"
)

func TestSweepThresholds(t *testing.T) {
	thresholds := SweepThresholds(0.01, 0.1, 0.02)

	want := []float32{0.01, 0.03, 0.05, 0.07, 0.09}
	if len(thresholds) != len(want) {
		t.Errorf("got %d thresholds, want %d", len(thresholds), len(want))
		t.Logf("got: %v", thresholds)
		return
	}

	for i := range want {
		diff := thresholds[i] - want[i]
		if diff < -0.001 || diff > 0.001 {
			t.Errorf("threshold[%d] = %v, want %v", i, thresholds[i], want[i])
		}
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./internal/bench/... -v -run TestSweepThresholds
```

Expected: FAIL (function not defined)

**Step 3: Write minimal implementation**

Create `internal/bench/sweep.go`:
```go
package bench

import (
	"context"
	"sort"

	sat "github.com/jamesainslie/go-sat"
)

// SweepResult holds metrics for one threshold value.
type SweepResult struct {
	Threshold float32
	Metrics   Metrics
}

// SweepThresholds generates threshold values from min to max with given step.
func SweepThresholds(min, max, step float32) []float32 {
	var thresholds []float32
	for t := min; t < max; t += step {
		thresholds = append(thresholds, t)
	}
	return thresholds
}

// Sweep evaluates multiple thresholds and returns results sorted by weighted score.
func Sweep(ctx context.Context, talks []*Talk, modelPath, tokenizerPath string, cfg Config, thresholds []float32) ([]SweepResult, error) {
	var results []SweepResult

	for _, threshold := range thresholds {
		seg, err := sat.New(modelPath, tokenizerPath, sat.WithThreshold(threshold))
		if err != nil {
			return nil, err
		}

		// Aggregate metrics across all talks
		var totalTP, totalFP, totalFN int
		for _, talk := range talks {
			cfg.Threshold = threshold
			m, err := EvaluateTalk(ctx, seg, talk, cfg)
			if err != nil {
				seg.Close()
				return nil, err
			}
			totalTP += m.TruePositives
			totalFP += m.FalsePositives
			totalFN += m.FalseNegatives
		}

		seg.Close()

		// Compute aggregate metrics
		agg := Metrics{
			TruePositives:  totalTP,
			FalsePositives: totalFP,
			FalseNegatives: totalFN,
		}
		if totalTP+totalFP > 0 {
			agg.Precision = float64(totalTP) / float64(totalTP+totalFP)
		}
		if totalTP+totalFN > 0 {
			agg.Recall = float64(totalTP) / float64(totalTP+totalFN)
		}
		if agg.Precision+agg.Recall > 0 {
			agg.F1 = 2 * agg.Precision * agg.Recall / (agg.Precision + agg.Recall)
		}
		wp := cfg.PrecisionWeight
		wr := cfg.RecallWeight
		if wp+wr > 0 {
			agg.WeightedScore = (wp*agg.Precision + wr*agg.Recall) / (wp + wr)
		}

		results = append(results, SweepResult{
			Threshold: threshold,
			Metrics:   agg,
		})
	}

	// Sort by weighted score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Metrics.WeightedScore > results[j].Metrics.WeightedScore
	})

	return results, nil
}
```

**Step 4: Run test to verify it passes**

```bash
go test ./internal/bench/... -v -run TestSweepThresholds
```

Expected: PASS

**Step 5: Commit**

```bash
git add internal/bench/sweep.go internal/bench/sweep_test.go
git commit -m "feat(bench): add threshold sweep with aggregated metrics"
```

---

## Task 11: Implement CLI Entry Point

**Files:**
- Create: `cmd/sat-bench/main.go`

**Step 1: Write CLI with flag parsing**

Create `cmd/sat-bench/main.go`:
```go
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	sat "github.com/jamesainslie/go-sat"
	"github.com/jamesainslie/go-sat/internal/bench"
)

func main() {
	var (
		modelPath     = flag.String("model", "", "Path to ONNX model file (required)")
		tokenizerPath = flag.String("tokenizer", "", "Path to tokenizer model file (required)")
		corpusDir     = flag.String("corpus", "testdata/ted", "Directory containing transcript files")
		threshold     = flag.Float64("threshold", 0.025, "Boundary detection threshold")
		tolerance     = flag.Int("tolerance", 3, "Character tolerance for boundary matching")
		wp            = flag.Float64("wp", 1.0, "Precision weight")
		wr            = flag.Float64("wr", 1.0, "Recall weight")
		sweep         = flag.Bool("sweep", false, "Run threshold sweep")
		sweepMin      = flag.Float64("sweep-min", 0.01, "Sweep minimum threshold")
		sweepMax      = flag.Float64("sweep-max", 0.20, "Sweep maximum threshold")
		sweepStep     = flag.Float64("sweep-step", 0.01, "Sweep step size")
		models        = flag.String("models", "", "Comma-separated model paths for comparison")
	)
	flag.Parse()

	if *modelPath == "" && *models == "" {
		fmt.Fprintln(os.Stderr, "error: -model or -models required")
		flag.Usage()
		os.Exit(1)
	}
	if *tokenizerPath == "" {
		fmt.Fprintln(os.Stderr, "error: -tokenizer required")
		flag.Usage()
		os.Exit(1)
	}

	// Load corpus
	talks, err := bench.LoadCorpus(*corpusDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading corpus: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d talks from %s\n\n", len(talks), *corpusDir)

	cfg := bench.Config{
		Threshold:       float32(*threshold),
		Tolerance:       *tolerance,
		PrecisionWeight: *wp,
		RecallWeight:    *wr,
	}

	ctx := context.Background()

	if *models != "" {
		// Model comparison mode
		modelPaths := strings.Split(*models, ",")
		runModelComparison(ctx, modelPaths, *tokenizerPath, talks, cfg, *sweep, float32(*sweepMin), float32(*sweepMax), float32(*sweepStep))
	} else if *sweep {
		// Single model sweep mode
		runSweep(ctx, *modelPath, *tokenizerPath, talks, cfg, float32(*sweepMin), float32(*sweepMax), float32(*sweepStep))
	} else {
		// Single threshold evaluation
		runSingle(ctx, *modelPath, *tokenizerPath, talks, cfg)
	}
}

func runSingle(ctx context.Context, modelPath, tokenizerPath string, talks []*bench.Talk, cfg bench.Config) {
	seg, err := sat.New(modelPath, tokenizerPath, sat.WithThreshold(cfg.Threshold))
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating segmenter: %v\n", err)
		os.Exit(1)
	}
	defer seg.Close()

	var totalTP, totalFP, totalFN int
	for _, talk := range talks {
		m, err := bench.EvaluateTalk(ctx, seg, talk, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error evaluating %s: %v\n", talk.ID, err)
			os.Exit(1)
		}
		totalTP += m.TruePositives
		totalFP += m.FalsePositives
		totalFN += m.FalseNegatives
	}

	printMetrics(totalTP, totalFP, totalFN, cfg)
}

func runSweep(ctx context.Context, modelPath, tokenizerPath string, talks []*bench.Talk, cfg bench.Config, min, max, step float32) {
	thresholds := bench.SweepThresholds(min, max, step)

	fmt.Printf("Threshold Sweep Results (wp=%.1f, wr=%.1f)\n", cfg.PrecisionWeight, cfg.RecallWeight)
	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("%-8s %-8s %-8s %-8s %-8s\n", "Thresh", "Prec", "Rec", "F1", "Weighted")

	results, err := bench.Sweep(ctx, talks, modelPath, tokenizerPath, cfg, thresholds)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error during sweep: %v\n", err)
		os.Exit(1)
	}

	// Print sorted by threshold for readability
	for _, t := range thresholds {
		for _, r := range results {
			if r.Threshold == t {
				fmt.Printf("%-8.3f %-8.2f %-8.2f %-8.2f %-8.2f\n",
					r.Threshold, r.Metrics.Precision, r.Metrics.Recall, r.Metrics.F1, r.Metrics.WeightedScore)
				break
			}
		}
	}

	fmt.Println(strings.Repeat("-", 50))
	if len(results) > 0 {
		best := results[0]
		fmt.Printf("Optimal: %.3f (Weighted: %.2f)\n", best.Threshold, best.Metrics.WeightedScore)
	}
}

func runModelComparison(ctx context.Context, modelPaths []string, tokenizerPath string, talks []*bench.Talk, cfg bench.Config, sweep bool, min, max, step float32) {
	fmt.Printf("Model Comparison (wp=%.1f, wr=%.1f)\n", cfg.PrecisionWeight, cfg.RecallWeight)
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("%-30s %-8s %-8s %-8s\n", "Model", "Thresh", "F1", "Weighted")

	for _, modelPath := range modelPaths {
		var bestThreshold float32
		var bestMetrics bench.Metrics

		if sweep {
			thresholds := bench.SweepThresholds(min, max, step)
			results, err := bench.Sweep(ctx, talks, modelPath, tokenizerPath, cfg, thresholds)
			if err != nil {
				fmt.Fprintf(os.Stderr, "error with %s: %v\n", modelPath, err)
				continue
			}
			if len(results) > 0 {
				bestThreshold = results[0].Threshold
				bestMetrics = results[0].Metrics
			}
		} else {
			seg, err := sat.New(modelPath, tokenizerPath, sat.WithThreshold(cfg.Threshold))
			if err != nil {
				fmt.Fprintf(os.Stderr, "error with %s: %v\n", modelPath, err)
				continue
			}
			var totalTP, totalFP, totalFN int
			for _, talk := range talks {
				m, _ := bench.EvaluateTalk(ctx, seg, talk, cfg)
				totalTP += m.TruePositives
				totalFP += m.FalsePositives
				totalFN += m.FalseNegatives
			}
			seg.Close()

			bestThreshold = cfg.Threshold
			bestMetrics = computeMetrics(totalTP, totalFP, totalFN, cfg)
		}

		fmt.Printf("%-30s %-8.3f %-8.2f %-8.2f\n", modelPath, bestThreshold, bestMetrics.F1, bestMetrics.WeightedScore)
	}
}

func printMetrics(tp, fp, fn int, cfg bench.Config) {
	m := computeMetrics(tp, fp, fn, cfg)
	fmt.Printf("Precision: %.2f  Recall: %.2f  F1: %.2f  Weighted: %.2f\n",
		m.Precision, m.Recall, m.F1, m.WeightedScore)
	fmt.Printf("(TP: %d, FP: %d, FN: %d)\n", tp, fp, fn)
}

func computeMetrics(tp, fp, fn int, cfg bench.Config) bench.Metrics {
	m := bench.Metrics{
		TruePositives:  tp,
		FalsePositives: fp,
		FalseNegatives: fn,
	}
	if tp+fp > 0 {
		m.Precision = float64(tp) / float64(tp+fp)
	}
	if tp+fn > 0 {
		m.Recall = float64(tp) / float64(tp+fn)
	}
	if m.Precision+m.Recall > 0 {
		m.F1 = 2 * m.Precision * m.Recall / (m.Precision + m.Recall)
	}
	if cfg.PrecisionWeight+cfg.RecallWeight > 0 {
		m.WeightedScore = (cfg.PrecisionWeight*m.Precision + cfg.RecallWeight*m.Recall) / (cfg.PrecisionWeight + cfg.RecallWeight)
	}
	return m
}
```

**Step 2: Build and test CLI**

```bash
go build -o sat-bench ./cmd/sat-bench
./sat-bench -h
```

Expected: Help output with all flags

**Step 3: Commit**

```bash
git add cmd/sat-bench/main.go
git commit -m "feat: add sat-bench CLI for benchmarking"
```

---

## Task 12: Integration Test with Real Data

**Files:**
- Manual testing

**Step 1: Run benchmark with real model**

```bash
./sat-bench \
  -model /path/to/model_optimized.onnx \
  -tokenizer /path/to/sentencepiece.bpe.model \
  -corpus testdata/ted
```

**Step 2: Run threshold sweep**

```bash
./sat-bench \
  -model /path/to/model_optimized.onnx \
  -tokenizer /path/to/sentencepiece.bpe.model \
  -corpus testdata/ted \
  -sweep
```

**Step 3: Verify output looks reasonable**

Check that:
- Precision/recall values are between 0 and 1
- Optimal threshold is identified
- No crashes or errors

**Step 4: Final commit with any fixes**

```bash
git add -A
git commit -m "test: verify benchmark suite with real model"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Create fixtures directory | testdata/ted/README.md |
| 2-3 | Add TED transcripts | testdata/ted/*.txt |
| 4 | Corpus header parsing | internal/bench/corpus.go |
| 5 | Sentence parsing | internal/bench/corpus.go |
| 6 | Talk loading | internal/bench/corpus.go |
| 7 | Corpus loading | internal/bench/corpus.go |
| 8 | Metrics calculation | internal/bench/metrics.go |
| 9 | Talk evaluation | internal/bench/metrics.go |
| 10 | Threshold sweep | internal/bench/sweep.go |
| 11 | CLI entry point | cmd/sat-bench/main.go |
| 12 | Integration test | Manual verification |

Total: ~12 tasks, each with TDD cycle and commit.
