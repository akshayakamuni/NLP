
from fractions import Fraction

# Features: free, win, offer
data = [
    ([1, 1, 1], "Spam"),
    ([1, 1, 0], "Spam"),
    ([1, 0, 1], "Spam"),
    ([0, 1, 1], "Spam"),
    ([0, 0, 1], "NotSpam"),
    ([0, 1, 0], "NotSpam"),
    ([1, 0, 0], "NotSpam")
]

email = [0, 0, 0]
classes = ["Spam", "NotSpam"]
scores = {}

for cls in classes:
    rows = [features for features, label in data if label == cls]
    prior = Fraction(len(rows), len(data))
    likelihood = Fraction(1, 1)

    for j, value in enumerate(email):
        count = sum(row[j] == value for row in rows)
        probability = Fraction(count, len(rows))
        likelihood *= probability

    scores[cls] = prior * likelihood

    print(cls)
    print("Prior:", prior, float(prior))
    print("Likelihood:", likelihood, float(likelihood))
    print("Score:", scores[cls], float(scores[cls]))

total = sum(scores.values())

print("\nPosterior probabilities:")
for cls in classes:
    print(cls, float(scores[cls] / total))

print("\nPrediction:", max(scores, key=scores.get))
