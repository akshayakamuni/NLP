
from collections import Counter, defaultdict

sentences = [
    ["I", "like", "NLP"],
    ["I", "like", "AI"],
    ["You", "like", "NLP"],
    ["You", "love", "AI"],
    ["We", "love", "NLP"]
]

d = 0.5
bigram_counts = Counter()
continuations = defaultdict(set)
distinct_bigrams = set()
history_counts = Counter()
history_continuations = defaultdict(set)

for sentence in sentences:
    for h, w in zip(sentence, sentence[1:]):
        bigram_counts[(h, w)] += 1
        history_counts[h] += 1
        history_continuations[h].add(w)
        continuations[w].add(h)
        distinct_bigrams.add((h, w))

total_distinct = len(distinct_bigrams)

def p_cont(word):
    return len(continuations[word]) / total_distinct

def p_kn(word, history):
    count_hw = bigram_counts[(history, word)]
    count_h = history_counts[history]
    distinct_after_h = len(history_continuations[history])

    first = max(count_hw - d, 0) / count_h
    lam = d * distinct_after_h / count_h
    second = lam * p_cont(word)

    print(f"c({history}, {word}) =", count_hw)
    print(f"c({history}) =", count_h)
    print("Distinct continuations =", distinct_after_h)
    print("P_cont =", p_cont(word))
    print("First term =", first)
    print("Interpolation weight =", lam)

    return first + second

print("P_KN(AI | love) =", p_kn("AI", "love"))
print()
print("P_KN(NLP | like) =", p_kn("NLP", "like"))
