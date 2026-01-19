#!/usr/bin/env python3
"""Generate golden test files for go-sat tokenizer validation."""

import json
from transformers import AutoTokenizer

def main():
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    test_cases = [
        "Hello",
        "Hello world",
        "Hello world.",
        "I want to",
        "Thank you very much.",
        "This is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "",  # empty string
        "café",  # non-ASCII
        "你好世界",  # Chinese
        "🎉",  # emoji
    ]

    results = []
    for text in test_cases:
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        results.append({
            "input": text,
            "token_ids": enc["input_ids"],
            "offsets": enc["offset_mapping"],
            "tokens": tok.convert_ids_to_tokens(enc["input_ids"]),
        })

    with open("testdata/tokenizer_golden.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(results)} test cases")

if __name__ == "__main__":
    main()
