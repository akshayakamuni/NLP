
import math
import numpy as np

sentences = [
    ["cat", "dog", "animal"],
    ["cat", "cat", "pet"],
    ["dog", "pet", "animal"],
    ["cat", "dog", "pet"]
]

vocabulary = ["cat", "dog", "animal", "pet"]
N = len(sentences)

df = {
    word: sum(word in sentence for sentence in sentences)
    for word in vocabulary
}

idf = {
    word: math.log(N / df[word])
    for word in vocabulary
}

vectors = []

for sentence in sentences:
    vector = [
        sentence.count(word) * idf[word]
        for word in vocabulary
    ]
    vectors.append(vector)

    print("Sentence:", " ".join(sentence))
    print("TF-IDF:", [round(x, 4) for x in vector])

best_pair = None
best_similarity = -1

for i in range(N):
    for j in range(i + 1, N):
        a = np.array(vectors[i])
        b = np.array(vectors[j])

        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        similarity = (
            float(np.dot(a, b) / denominator)
            if denominator else 0.0
        )

        print(f"S{i+1}, S{j+1}: {similarity:.4f}")

        if similarity > best_similarity:
            best_similarity = similarity
            best_pair = (i + 1, j + 1)

print("Closest pair:", best_pair)
print("Cosine similarity:", round(best_similarity, 4))
