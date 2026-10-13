package phonmatchnet

import (
	"errors"
	"testing"
)

func TestNormalizeKeywordMatchesNaptrap(t *testing.T) {
	cases := loadReference(t).Normalization
	if len(cases) == 0 {
		t.Fatal("no normalization fixtures")
	}
	for _, c := range cases {
		got, err := NormalizeKeyword(c.Input)
		if c.Error != "" {
			if !errors.Is(err, ErrInvalidKeyword) {
				t.Errorf("%q: got %q, %v; naptrap rejects it: %s", c.Input, got, err, c.Error)
			}
			continue
		}
		if err != nil || got != c.Normalized {
			t.Errorf("%q: got %q, %v; want %q", c.Input, got, err, c.Normalized)
		}
	}
	if _, err := NormalizeKeyword(string([]byte{0xff})); !errors.Is(err, ErrInvalidKeyword) {
		t.Errorf("invalid UTF-8: %v", err)
	}
}
