"""Quick local test for regex tokenizers (no dataset download required)."""

from tokenizer import sentence_tokenize, word_tokenize, tokenize_paragraph

SAMPLE = """
ఈ రోజు వాతావరణం చాలా మంచిగా ఉంది. నేను https://example.com/open చూశాను.
మా ఇమెయిల్ contact@test.org. ధర Rs. 99.50. తేదీ 05/08/2026.
అద్భుతం! మీరు ఎలా ఉన్నారు? తెలుగు భాష చాలా అందం.
"""

if __name__ == "__main__":
    print("--- Sentence Tokenization ---")
    for i, sent in enumerate(sentence_tokenize(SAMPLE.strip()), 1):
        print(f"{i}. {sent}")

    print("\n--- Word Tokenization (per sentence) ---")
    for sent_tokens in tokenize_paragraph(SAMPLE.strip()):
        print(" ".join(sent_tokens))
