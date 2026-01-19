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
