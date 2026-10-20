
from collections import Counter

corpus = {
    "low": 5,
    "lower": 2,
    "lowest": 3,
    "new": 4,
    "newer": 2
}

words = {word: list(word) for word in corpus}

for merge_number in range(2):
    pair_counts = Counter()

    for word, tokens in words.items():
        for a, b in zip(tokens, tokens[1:]):
            pair_counts[(a, b)] += corpus[word]

    if not pair_counts:
        break

    # Deterministic tie-breaking: first pair in sorted order.
    best_pair = sorted(
        pair_counts,
        key=lambda pair: (-pair_counts[pair], pair)
    )[0]

    merged = "".join(best_pair)
    print("Merge", merge_number + 1, ":", best_pair, "->", merged)

    for word, tokens in words.items():
        updated = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and (tokens[i], tokens[i + 1]) == best_pair
            ):
                updated.append(merged)
                i += 2
            else:
                updated.append(tokens[i])
                i += 1

        words[word] = updated

print("Tokenization of newest:", list("newest"))
