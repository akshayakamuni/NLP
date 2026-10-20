
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


# -------------------------------------------------
# STEP 1: File paths
# -------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent

UNIGRAM_FILE = DATA_DIR / "count_unigram.txt"
EDIT_FILE = DATA_DIR / "count_1edit.txt"
ERROR_FILE = DATA_DIR / "spell-errors.txt"


# -------------------------------------------------
# STEP 2: Load unigram word frequencies
# -------------------------------------------------

word_counts = {}

with open(UNIGRAM_FILE, encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()

        if len(parts) < 2:
            continue

        word = parts[0].lower()

        try:
            count = int(parts[1])
        except ValueError:
            continue

        word_counts[word] = count

total_word_count = sum(word_counts.values())

print("Unique words loaded:", len(word_counts))
print("Total unigram frequency:", total_word_count)


# -------------------------------------------------
# STEP 3: Load edit-operation frequencies
# -------------------------------------------------

edit_counts = Counter()

with open(EDIT_FILE, encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()

        if len(parts) < 2:
            continue

        operation = parts[0]

        try:
            count = int(parts[1])
        except ValueError:
            continue

        edit_counts[operation] += count

total_edit_count = sum(edit_counts.values())

print("Edit operations loaded:", len(edit_counts))


# -------------------------------------------------
# STEP 4: Convert a single edit into an operation
# -------------------------------------------------

def get_edit_operation(correct, observed):
    """
    Represent a one-edit transformation from the
    correct word to the observed misspelling.

    Examples:
        raining -> rainning
        cat -> cut
    """

    matcher = SequenceMatcher(
        None, correct, observed, autojunk=False
    )

    changes = [
        (tag, correct[i1:i2], observed[j1:j2])
        for tag, i1, i2, j1, j2
        in matcher.get_opcodes()
        if tag != "equal"
    ]

    if len(changes) != 1:
        return None

    tag, old, new = changes[0]

    # Multi-character replacement/insertion/deletion
    # segments are retained as a single operation.
    if tag == "insert":
        old = "-"
    elif tag == "delete":
        new = "-"

    return f"{old}|{new}"


# -------------------------------------------------
# STEP 5: Generate candidate words
# -------------------------------------------------

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"

def generate_candidates(observed):
    observed = observed.lower()
    candidates = set()

    # Candidate can be identical to observed if it
    # is already in the vocabulary.
    if observed in word_counts:
        candidates.add(observed)

    # Deletion: remove one character.
    for i in range(len(observed)):
        candidates.add(observed[:i] + observed[i + 1:])

    # Transposition: swap adjacent characters.
    for i in range(len(observed) - 1):
        candidate = (
            observed[:i]
            + observed[i + 1]
            + observed[i]
            + observed[i + 2:]
        )
        candidates.add(candidate)

    # Replacement: replace one character.
    for i in range(len(observed)):
        for char in ALPHABET:
            candidate = (
                observed[:i] + char + observed[i + 1:]
            )
            candidates.add(candidate)

    # Insertion: insert one character.
    for i in range(len(observed) + 1):
        for char in ALPHABET:
            candidate = (
                observed[:i] + char + observed[i:]
            )
            candidates.add(candidate)

    # Retain candidates present in the vocabulary.
    return {
        candidate for candidate in candidates
        if candidate in word_counts
    }


# -------------------------------------------------
# STEP 6: Noisy-channel scoring
# -------------------------------------------------

def score_candidate(correct, observed):
    """
    Score P(observed | correct) * P(correct).

    Log probabilities avoid numerical underflow.
    """

    if total_word_count == 0:
        return float("-inf")

    word_probability = (
        word_counts[correct] / total_word_count
    )

    if word_probability <= 0:
        return float("-inf")

    operation = get_edit_operation(correct, observed)

    # If the words are identical, use no edit-channel
    # factor; the language prior determines the score.
    if correct == observed:
        return math.log(word_probability)

    if operation is None:
        return float("-inf")

    operation_count = edit_counts.get(operation, 0)

    # Unknown or unobserved operations receive a small
    # floor instead of making every candidate impossible.
    channel_probability = (
        operation_count / total_edit_count
        if total_edit_count and operation_count > 0
        else 1e-12
    )

    return (
        math.log(word_probability)
        + math.log(channel_probability)
    )


# -------------------------------------------------
# STEP 7: Correct one word
# -------------------------------------------------

def correct_word(observed):
    observed = observed.lower().strip()

    candidates = generate_candidates(observed)

    if not candidates:
        return observed

    return max(
        candidates,
        key=lambda candidate: score_candidate(
            candidate, observed
        )
    )


# -------------------------------------------------
# STEP 8: Load annotated spelling errors
# -------------------------------------------------

def load_test_data():
    test_data = []

    with open(ERROR_FILE, encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue

            correct, errors = line.split(":", 1)
            correct = correct.strip().lower()

            if correct not in word_counts:
                continue

            for item in errors.split(","):
                item = item.strip().lower()

                # The dataset uses suffixes such as *5
                # to indicate a repeated error count.
                item = re.sub(r"\*\d+$", "", item)

                if not item:
                    continue

                # Evaluate only genuine misspellings
                # one edit away from the correct word.
                if (
                    item != correct
                    and get_edit_operation(correct, item)
                    is not None
                    and len(item) > 0
                ):
                    test_data.append((item, correct))

    return test_data


# -------------------------------------------------
# STEP 9: Evaluate accuracy
# -------------------------------------------------

def evaluate():
    test_data = load_test_data()

    if not test_data:
        print("No eligible test examples found.")
        return

    correct_predictions = 0
    results = []

    for observed, expected in test_data:
        predicted = correct_word(observed)
        is_correct = predicted == expected

        if is_correct:
            correct_predictions += 1

        results.append(
            (observed, expected, predicted, is_correct)
        )

    accuracy = 100 * correct_predictions / len(test_data)

    print("\nTotal test examples:", len(test_data))
    print("Correct predictions:", correct_predictions)
    print(f"Accuracy: {accuracy:.2f}%")

    print("\nSample predictions:")
    for observed, expected, predicted, is_correct in results[:20]:
        print(
            f"{observed:20s} | expected={expected:15s} "
            f"| predicted={predicted:15s} "
            f"| {'CORRECT' if is_correct else 'WRONG'}"
        )


# -------------------------------------------------
# STEP 10: Run evaluation and interactive testing
# -------------------------------------------------

if __name__ == "__main__":
    evaluate()

    print("\nTry your own misspellings.")
    print("Type 'exit' to stop.")

    while True:
        observed = input("Misspelled word: ").strip()

        if observed.lower() == "exit":
            break

        print("Correction:", correct_word(observed))
