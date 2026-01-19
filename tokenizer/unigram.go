package tokenizer

const negInf = -1e9

// EncodeIDs returns HuggingFace-compatible token IDs for the input text.
func (t *Tokenizer) EncodeIDs(text string) []int32 {
	tokens := t.Encode(text)
	ids := make([]int32, len(tokens))
	for i, tok := range tokens {
		ids[i] = tok.ID
	}
	return ids
}

// Encode tokenizes text using Viterbi algorithm, returning tokens with offsets.
func (t *Tokenizer) Encode(text string) []TokenInfo {
	if text == "" {
		return nil
	}

	// Normalize text (add ▁ prefix, replace spaces)
	normalized := normalize(text)
	if normalized == "" {
		return nil
	}

	runes := []rune(normalized)
	n := len(runes)

	// best[i] = best log probability to tokenize runes[0:i]
	best := make([]float64, n+1)
	// parent[i] = start position of the token ending at position i
	parent := make([]int, n+1)
	// tokenAt[i] = the token string ending at position i
	tokenAt := make([]string, n+1)

	for i := 1; i <= n; i++ {
		best[i] = negInf
		parent[i] = -1
	}

	// Dynamic programming: find best tokenization
	for i := 1; i <= n; i++ {
		// Try all possible tokens ending at position i
		maxLen := t.maxTokenLen
		if maxLen > i {
			maxLen = i
		}

		for length := 1; length <= maxLen; length++ {
			j := i - length
			substr := string(runes[j:i])

			score, exists := t.scores[substr]
			if !exists {
				continue
			}

			candidate := best[j] + float64(score)
			if candidate > best[i] {
				best[i] = candidate
				parent[i] = j
				tokenAt[i] = substr
			}
		}

		// If no valid token found, use unknown token for single character
		if best[i] == negInf {
			// Use <unk> token score (convert HF ID to SentencePiece index for lookup)
			unkSPIndex := t.hfIDToSPIndex(t.unkID)
			best[i] = best[i-1] + float64(t.scores[t.idToPiece[unkSPIndex]])
			parent[i] = i - 1
			tokenAt[i] = string(runes[i-1 : i])
		}
	}

	// Backtrack to get tokens
	var tokens []TokenInfo
	pos := n
	for pos > 0 {
		start := parent[pos]
		tokenStr := tokenAt[pos]

		spIndex, ok := t.pieces[tokenStr]
		if !ok {
			spIndex = 0 // <unk> is at SentencePiece index 0
		}

		// Convert SentencePiece index to HuggingFace ID
		hfID := t.spIndexToHFID(spIndex)

		tokens = append(tokens, TokenInfo{
			ID:    hfID,
			Text:  tokenStr,
			Start: start,
			End:   pos,
		})
		pos = start
	}

	// Reverse to get correct order
	for i, j := 0, len(tokens)-1; i < j; i, j = i+1, j-1 {
		tokens[i], tokens[j] = tokens[j], tokens[i]
	}

	return tokens
}
