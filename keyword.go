package phonmatchnet

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/text/unicode/norm"
)

// keywordTokens matches naptrap's training-time tokenization.
var keywordTokens = regexp.MustCompile(`[a-z]+(?:'[a-z]+)*|[0-9]+`)

// Keyword holds a keyword's precomputed text features. Create it with
// Engine.Keyword and reuse it across clips. The zero Keyword is invalid.
type Keyword struct {
	text     string
	features []float32 // positions x hiddenSize, row-major; never mutated
}

// Text returns the normalized keyword.
func (k Keyword) Text() string { return k.text }

// Positions returns the number of text feature rows.
func (k Keyword) Positions() int { return len(k.features) / hiddenSize }

// NormalizeKeyword applies the policy used during training: NFKD, accents and
// punctuation removed, lowercase, words joined by single spaces. Apostrophes
// inside words are kept. Numbers must be spelled out.
func NormalizeKeyword(text string) (string, error) {
	if !utf8.ValidString(text) {
		return "", fmt.Errorf("%w: keyword must be valid UTF-8", ErrInvalidKeyword)
	}
	text = norm.NFKD.String(strings.ReplaceAll(text, "’", "'"))
	var b strings.Builder
	for _, r := range text {
		if norm.NFKD.PropertiesString(string(r)).CCC() == 0 {
			b.WriteRune(r)
		}
	}
	words := keywordTokens.FindAllString(strings.ToLower(b.String()), -1)
	for _, word := range words {
		if word[0] >= '0' && word[0] <= '9' {
			return "", fmt.Errorf("%w: spell out numbers in the keyword, e.g. 'route sixty six'", ErrInvalidKeyword)
		}
	}
	if len(words) == 0 {
		return "", fmt.Errorf("%w: keyword contains no English letters", ErrInvalidKeyword)
	}
	return strings.Join(words, " "), nil
}

// Keyword normalizes text and computes its g2p decoder features. Each word is
// encoded, then decoded greedily; every decoder state before the end token is
// one feature row, and a separator row sits between words.
func (e *Engine) Keyword(ctx context.Context, text string) (Keyword, error) {
	normalized, err := NormalizeKeyword(text)
	if err != nil {
		return Keyword{}, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return Keyword{}, ErrClosed
	}
	var features []float32
	for i, word := range strings.Fields(normalized) {
		if err := ctx.Err(); err != nil {
			return Keyword{}, err
		}
		if features, err = e.encodeWord(ctx, word, i > 0, features); err != nil {
			return Keyword{}, err
		}
		if len(features) > MaxKeywordPositions*hiddenSize {
			break
		}
	}
	positions := len(features) / hiddenSize
	if positions == 0 || positions > MaxKeywordPositions {
		return Keyword{}, fmt.Errorf("%w: %q produced %d text positions; 1 to %d are supported",
			ErrInvalidKeyword, normalized, positions, MaxKeywordPositions)
	}
	return Keyword{text: normalized, features: features}, nil
}

// encodeWord appends one word's decoder states to features, preceded by the
// separator row when separate is set.
func (e *Engine) encodeWord(ctx context.Context, word string, separate bool, features []float32) ([]float32, error) {
	ids := make([]int64, 0, len(word)+1)
	for _, c := range word {
		id, ok := e.graphemes[c]
		if !ok {
			id = e.unknownGrapheme
		}
		ids = append(ids, id)
	}
	ids = append(ids, e.endGrapheme)
	graphemes, err := ort.NewTensor(ort.NewShape(int64(len(ids))), ids)
	if err != nil {
		return nil, err
	}
	encoded, err := run(e.g2pEncoder, []ort.Value{graphemes}, 2)
	_ = graphemes.Destroy()
	if err != nil {
		return nil, fmt.Errorf("g2p encoder: %w", err)
	}
	hidden, separator := encoded[0], encoded[1]
	defer func() { _ = separator.Destroy() }()
	defer func() { _ = hidden.Destroy() }()
	if separate {
		row, err := floats(separator, hiddenSize)
		if err != nil {
			return nil, fmt.Errorf("g2p separator: %w", err)
		}
		features = append(features, row...)
	}
	token := []int64{e.g2p.StartToken}
	input, err := ort.NewTensor(ort.NewShape(1), token)
	if err != nil {
		return nil, err
	}
	defer func() { _ = input.Destroy() }()
	for range e.g2p.MaxDecoderSteps {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		decoded, err := run(e.g2pDecoder, []ort.Value{input, hidden}, 2)
		if err != nil {
			return nil, fmt.Errorf("g2p decoder: %w", err)
		}
		_ = hidden.Destroy()
		hidden = decoded[0]
		logits, err := floats(decoded[1], e.g2p.Phonemes)
		_ = decoded[1].Destroy()
		if err != nil {
			return nil, fmt.Errorf("g2p decoder logits: %w", err)
		}
		// First maximum, like torch.argmax.
		best := 0
		for i, value := range logits {
			if value > logits[best] {
				best = i
			}
		}
		if int64(best) == e.g2p.EndToken {
			break
		}
		row, err := floats(hidden, hiddenSize)
		if err != nil {
			return nil, fmt.Errorf("g2p decoder state: %w", err)
		}
		features = append(features, row...)
		input.GetData()[0] = int64(best)
	}
	return features, nil
}

// floats returns a copy of a float32 output with the expected element count,
// rejecting nonfinite values.
func floats(value ort.Value, count int) ([]float32, error) {
	tensor, ok := value.(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("output must be a float32 tensor")
	}
	data := tensor.GetData()
	if len(data) != count {
		return nil, fmt.Errorf("output has %d values, expected %d", len(data), count)
	}
	if !finite(data) {
		return nil, errors.New("output contains nonfinite values")
	}
	return append([]float32(nil), data...), nil
}
