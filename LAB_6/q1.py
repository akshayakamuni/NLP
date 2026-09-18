# ============================================================
# NLP LAB ASSIGNMENT 2
# N-GRAM LANGUAGE MODELS WITH SMOOTHING TECHNIQUES
# ============================================================
# Uses the same dataset path from the previous code.
#
# Models:
#   Unigram, Bigram, Trigram, Quadgram
#
# Smoothing:
#   1. Interpolated Smoothing
#   2. Good-Turing Smoothing
#   3. Katz Backoff Smoothing
#   4. Stupid Backoff Smoothing
#   5. Kneser-Ney Smoothing
#
# IMPORTANT:
# Reduced dataset for faster execution:
#   Total      = 10,000 sentences
#   Training   = 8,000
#   Development= 1,000
#   Test       = 1,000
# ============================================================

from collections import Counter
from pathlib import Path
import math
import random


# ============================================================
# 1. CONFIGURATION
# ============================================================

FILE_NAME = (
    Path(__file__).resolve().parent.parent
    / "LAB_4"
    / "indiccorp_te_tokenized.txt"
)

# REDUCED SIZE FOR FAST EXECUTION
TOTAL_SENTENCES = 10000
TRAIN_SIZE = 8000
DEV_SIZE = 1000
TEST_SIZE = 1000

RANDOM_SEED = 42

BIGRAM_LAMBDAS = [0.3, 0.7]
TRIGRAM_LAMBDAS = [0.1, 0.3, 0.6]
QUADGRAM_LAMBDAS = [0.05, 0.15, 0.30, 0.50]

DISCOUNT = 0.75
STUPID_BACKOFF_ALPHA = 0.4
EPSILON = 1e-12

random.seed(RANDOM_SEED)


# ============================================================
# 2. LOAD SENTENCES
# ============================================================

def load_sentences(filename, limit):
    sentences = []

    print("\nReading dataset...")
    print("File:", filename)

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            tokens = line.split()

            if tokens:
                sentences.append(tokens)

            if len(sentences) >= limit:
                break

    return sentences


# ============================================================
# 3. REPLACE OOV WORDS
# ============================================================

def replace_unknown_words(sentences, vocabulary):
    return [
        [word if word in vocabulary else "<UNK>" for word in sentence]
        for sentence in sentences
    ]


# ============================================================
# 4. BUILD N-GRAM MODEL
# ============================================================

def build_ngram_model(sentences, n):
    ngram_counts = Counter()
    context_counts = Counter()

    for sentence in sentences:

        if n == 1:
            tokens = sentence + ["</s>"]
        else:
            tokens = ["<s>"] * (n - 1) + sentence + ["</s>"]

        for i in range(len(tokens) - n + 1):

            ngram = tuple(tokens[i:i + n])
            ngram_counts[ngram] += 1

            if n > 1:
                context = tuple(tokens[i:i + n - 1])
                context_counts[context] += 1

    return {
        "ngram_counts": ngram_counts,
        "context_counts": context_counts,
        "total_tokens": sum(ngram_counts.values())
    }


# ============================================================
# 5. TRAIN FOUR MODELS
# ============================================================

def train_models(training_data):
    models = {}

    print("\n" + "=" * 70)
    print("TRAINING N-GRAM MODELS")
    print("=" * 70)

    for n in range(1, 5):

        names = {
            1: "UNIGRAM",
            2: "BIGRAM",
            3: "TRIGRAM",
            4: "QUADGRAM"
        }

        print(f"\nTraining {names[n]}...")

        models[n] = build_ngram_model(training_data, n)

        print("Unique n-grams:", len(models[n]["ngram_counts"]))

    return models


# ============================================================
# 6. BASIC MLE PROBABILITIES
# ============================================================

def unigram_probability(word, models):
    model = models[1]
    count = model["ngram_counts"].get((word,), 0)

    return count / model["total_tokens"] if model["total_tokens"] else EPSILON


def bigram_probability(word, history, models):
    if not history:
        return unigram_probability(word, models)

    model = models[2]
    context = (history[-1],)
    count = model["ngram_counts"].get(context + (word,), 0)
    denominator = model["context_counts"].get(context, 0)

    if denominator == 0:
        return 0.0

    return count / denominator


def trigram_probability(word, history, models):
    if len(history) < 2:
        return bigram_probability(word, history, models)

    model = models[3]
    context = tuple(history[-2:])
    count = model["ngram_counts"].get(context + (word,), 0)
    denominator = model["context_counts"].get(context, 0)

    if denominator == 0:
        return 0.0

    return count / denominator


def quadgram_probability(word, history, models):
    if len(history) < 3:
        return trigram_probability(word, history, models)

    model = models[4]
    context = tuple(history[-3:])
    count = model["ngram_counts"].get(context + (word,), 0)
    denominator = model["context_counts"].get(context, 0)

    if denominator == 0:
        return 0.0

    return count / denominator


# ============================================================
# 7. INTERPOLATED SMOOTHING
# ============================================================

def interpolated_bigram(word, history, models):
    l1, l2 = BIGRAM_LAMBDAS

    return (
        l1 * unigram_probability(word, models)
        + l2 * bigram_probability(word, history, models)
    )


def interpolated_trigram(word, history, models):
    l1, l2, l3 = TRIGRAM_LAMBDAS

    return (
        l1 * unigram_probability(word, models)
        + l2 * bigram_probability(word, history, models)
        + l3 * trigram_probability(word, history, models)
    )


def interpolated_quadgram(word, history, models):
    l1, l2, l3, l4 = QUADGRAM_LAMBDAS

    return (
        l1 * unigram_probability(word, models)
        + l2 * bigram_probability(word, history, models)
        + l3 * trigram_probability(word, history, models)
        + l4 * quadgram_probability(word, history, models)
    )


# ============================================================
# 8. GOOD-TURING
# ============================================================

def create_good_turing_table(counts):
    freq_of_freq = Counter(counts.values())
    adjusted = {}

    for ngram, r in counts.items():

        nr = freq_of_freq.get(r, 0)
        nr1 = freq_of_freq.get(r + 1, 0)

        if nr > 0 and nr1 > 0:
            r_star = (r + 1) * nr1 / nr
        else:
            r_star = r

        adjusted[ngram] = r_star

    return adjusted


def good_turing_probability(word, history, n, models, gt_tables):
    model = models[n]
    counts = model["ngram_counts"]

    if n == 1:
        ngram = (word,)
        r = counts.get(ngram, 0)

        if r == 0:
            return EPSILON

        return max(
            gt_tables[1].get(ngram, r) / model["total_tokens"],
            EPSILON
        )

    if n == 2:
        if not history:
            return unigram_probability(word, models)
        context = (history[-1],)

    elif n == 3:
        if len(history) < 2:
            return good_turing_probability(
                word, history, 2, models, gt_tables
            )
        context = tuple(history[-2:])

    else:
        if len(history) < 3:
            return good_turing_probability(
                word, history, 3, models, gt_tables
            )
        context = tuple(history[-3:])

    ngram = context + (word,)
    r = counts.get(ngram, 0)
    denominator = model["context_counts"].get(context, 0)

    if denominator == 0:
        return unigram_probability(word, models)

    if r == 0:
        return EPSILON

    r_star = gt_tables[n].get(ngram, r)

    return max(r_star / denominator, EPSILON)


# ============================================================
# 9. KATZ BACKOFF
# ============================================================

def katz_bigram(word, history, models):
    if history:
        context = (history[-1],)
        ngram = context + (word,)

        if models[2]["ngram_counts"].get(ngram, 0) > 0:
            denominator = models[2]["context_counts"].get(context, 0)

            if denominator:
                return models[2]["ngram_counts"][ngram] / denominator

    return max(unigram_probability(word, models), EPSILON)


def katz_trigram(word, history, models):
    if len(history) >= 2:
        context = tuple(history[-2:])
        ngram = context + (word,)

        if models[3]["ngram_counts"].get(ngram, 0) > 0:
            denominator = models[3]["context_counts"].get(context, 0)

            if denominator:
                return models[3]["ngram_counts"][ngram] / denominator

    return max(katz_bigram(word, history, models), EPSILON)


def katz_quadgram(word, history, models):
    if len(history) >= 3:
        context = tuple(history[-3:])
        ngram = context + (word,)

        if models[4]["ngram_counts"].get(ngram, 0) > 0:
            denominator = models[4]["context_counts"].get(context, 0)

            if denominator:
                return models[4]["ngram_counts"][ngram] / denominator

    return max(katz_trigram(word, history, models), EPSILON)


# ============================================================
# 10. STUPID BACKOFF
# ============================================================

def stupid_bigram(word, history, models):
    if history:
        context = (history[-1],)
        ngram = context + (word,)

        count = models[2]["ngram_counts"].get(ngram, 0)
        denominator = models[2]["context_counts"].get(context, 0)

        if count > 0 and denominator > 0:
            return count / denominator

    return STUPID_BACKOFF_ALPHA * unigram_probability(word, models)


def stupid_trigram(word, history, models):
    if len(history) >= 2:
        context = tuple(history[-2:])
        ngram = context + (word,)

        count = models[3]["ngram_counts"].get(ngram, 0)
        denominator = models[3]["context_counts"].get(context, 0)

        if count > 0 and denominator > 0:
            return count / denominator

    if len(history) >= 1:
        context = (history[-1],)
        ngram = context + (word,)

        count = models[2]["ngram_counts"].get(ngram, 0)
        denominator = models[2]["context_counts"].get(context, 0)

        if count > 0 and denominator > 0:
            return STUPID_BACKOFF_ALPHA * count / denominator

    return STUPID_BACKOFF_ALPHA ** 2 * unigram_probability(word, models)


def stupid_quadgram(word, history, models):
    if len(history) >= 3:
        context = tuple(history[-3:])
        ngram = context + (word,)

        count = models[4]["ngram_counts"].get(ngram, 0)
        denominator = models[4]["context_counts"].get(context, 0)

        if count > 0 and denominator > 0:
            return count / denominator

    if len(history) >= 2:
        context = tuple(history[-2:])
        ngram = context + (word,)

        count = models[3]["ngram_counts"].get(ngram, 0)
        denominator = models[3]["context_counts"].get(context, 0)

        if count > 0 and denominator > 0:
            return STUPID_BACKOFF_ALPHA * count / denominator

    if len(history) >= 1:
        context = (history[-1],)
        ngram = context + (word,)

        count = models[2]["ngram_counts"].get(ngram, 0)
        denominator = models[2]["context_counts"].get(context, 0)

        if count > 0 and denominator > 0:
            return STUPID_BACKOFF_ALPHA ** 2 * count / denominator

    return STUPID_BACKOFF_ALPHA ** 3 * unigram_probability(word, models)


# ============================================================
# 11. PRECOMPUTE KNESER-NEY STATISTICS
# ============================================================

def prepare_kneser_ney(models):
    bigrams = models[2]["ngram_counts"]
    trigrams = models[3]["ngram_counts"]
    quadgrams = models[4]["ngram_counts"]

    # Unique preceding words for each word
    preceding = Counter()

    # Unique following words for each context
    following_bigram = Counter()
    following_trigram = Counter()
    following_quadgram = Counter()

    for a, b in bigrams:
        preceding[b] += 1
        following_bigram[(a,)] += 1

    for a, b, c in trigrams:
        following_trigram[(a, b)] += 1

    for a, b, c, d in quadgrams:
        following_quadgram[(a, b, c)] += 1

    total_unique_bigrams = len(bigrams)

    return {
        "preceding": preceding,
        "following_bigram": following_bigram,
        "following_trigram": following_trigram,
        "following_quadgram": following_quadgram,
        "total_unique_bigrams": total_unique_bigrams
    }


def continuation_probability(word, kn):
    if kn["total_unique_bigrams"] == 0:
        return EPSILON

    return (
        kn["preceding"].get(word, 0)
        / kn["total_unique_bigrams"]
    )


# ============================================================
# 12. KNESER-NEY
# ============================================================

def kneser_ney_bigram(word, history, models, kn):
    if not history:
        return continuation_probability(word, kn)

    previous = history[-1]
    context = (previous,)
    ngram = context + (word,)

    count = models[2]["ngram_counts"].get(ngram, 0)
    context_count = models[2]["context_counts"].get(context, 0)

    if context_count == 0:
        return continuation_probability(word, kn)

    unique_following = kn["following_bigram"].get(context, 0)

    lambda_value = (
        DISCOUNT * unique_following / context_count
    )

    first_term = max(count - DISCOUNT, 0) / context_count

    return max(
        first_term
        + lambda_value * continuation_probability(word, kn),
        EPSILON
    )


def kneser_ney_trigram(word, history, models, kn):
    if len(history) < 2:
        return kneser_ney_bigram(word, history, models, kn)

    context = tuple(history[-2:])
    ngram = context + (word,)

    count = models[3]["ngram_counts"].get(ngram, 0)
    context_count = models[3]["context_counts"].get(context, 0)

    if context_count == 0:
        return kneser_ney_bigram(
            word, history[-1:], models, kn
        )

    unique_following = kn["following_trigram"].get(context, 0)

    lambda_value = (
        DISCOUNT * unique_following / context_count
    )

    first_term = max(count - DISCOUNT, 0) / context_count

    return max(
        first_term
        + lambda_value * kneser_ney_bigram(
            word, history[-1:], models, kn
        ),
        EPSILON
    )


def kneser_ney_quadgram(word, history, models, kn):
    if len(history) < 3:
        return kneser_ney_trigram(
            word, history, models, kn
        )

    context = tuple(history[-3:])
    ngram = context + (word,)

    count = models[4]["ngram_counts"].get(ngram, 0)
    context_count = models[4]["context_counts"].get(context, 0)

    if context_count == 0:
        return kneser_ney_trigram(
            word, history[-2:], models, kn
        )

    unique_following = kn["following_quadgram"].get(context, 0)

    lambda_value = (
        DISCOUNT * unique_following / context_count
    )

    first_term = max(count - DISCOUNT, 0) / context_count

    return max(
        first_term
        + lambda_value * kneser_ney_trigram(
            word, history[-2:], models, kn
        ),
        EPSILON
    )


# ============================================================
# 13. PREPARE SENTENCE
# ============================================================

def prepare_sentence(sentence, n):
    if n == 1:
        return sentence + ["</s>"]

    return ["<s>"] * (n - 1) + sentence + ["</s>"]


# ============================================================
# 14. PERPLEXITY
# ============================================================

def calculate_perplexity(sentences, n, probability_function):
    total_log_probability = 0.0
    total_words = 0

    for sentence in sentences:
        tokens = prepare_sentence(sentence, n)

        for i in range(n - 1, len(tokens)):
            word = tokens[i]

            if n == 1:
                history = []
            else:
                history = tokens[max(0, i - n + 1):i]

            probability = probability_function(word, history)
            probability = max(probability, EPSILON)

            total_log_probability += math.log(probability)
            total_words += 1

    if total_words == 0:
        return float("inf")

    return math.exp(
        -total_log_probability / total_words
    )


# ============================================================
# 15. CREATE SMOOTHING MODELS
# ============================================================

def create_smoothing_models(models, gt_tables, kn):
    return {
        "Interpolated Bigram": (
            2,
            lambda w, h: interpolated_bigram(w, h, models)
        ),

        "Interpolated Trigram": (
            3,
            lambda w, h: interpolated_trigram(w, h, models)
        ),

        "Interpolated Quadgram": (
            4,
            lambda w, h: interpolated_quadgram(w, h, models)
        ),

        "Good-Turing Bigram": (
            2,
            lambda w, h: good_turing_probability(
                w, h, 2, models, gt_tables
            )
        ),

        "Good-Turing Trigram": (
            3,
            lambda w, h: good_turing_probability(
                w, h, 3, models, gt_tables
            )
        ),

        "Good-Turing Quadgram": (
            4,
            lambda w, h: good_turing_probability(
                w, h, 4, models, gt_tables
            )
        ),

        "Katz Bigram": (
            2,
            lambda w, h: katz_bigram(w, h, models)
        ),

        "Katz Trigram": (
            3,
            lambda w, h: katz_trigram(w, h, models)
        ),

        "Katz Quadgram": (
            4,
            lambda w, h: katz_quadgram(w, h, models)
        ),

        "Stupid Backoff Bigram": (
            2,
            lambda w, h: stupid_bigram(w, h, models)
        ),

        "Stupid Backoff Trigram": (
            3,
            lambda w, h: stupid_trigram(w, h, models)
        ),

        "Stupid Backoff Quadgram": (
            4,
            lambda w, h: stupid_quadgram(w, h, models)
        ),

        "Kneser-Ney Bigram": (
            2,
            lambda w, h: kneser_ney_bigram(
                w, h, models, kn
            )
        ),

        "Kneser-Ney Trigram": (
            3,
            lambda w, h: kneser_ney_trigram(
                w, h, models, kn
            )
        ),

        "Kneser-Ney Quadgram": (
            4,
            lambda w, h: kneser_ney_quadgram(
                w, h, models, kn
            )
        )
    }


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("N-GRAM LANGUAGE MODEL WITH SMOOTHING")
    print("=" * 80)

    print("\nUsing reduced dataset for faster execution.")
    print("Total sentences:", TOTAL_SENTENCES)
    print("Training:", TRAIN_SIZE)
    print("Development:", DEV_SIZE)
    print("Test:", TEST_SIZE)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    sentences = load_sentences(
        FILE_NAME,
        TOTAL_SENTENCES
    )

    print("\nSentences loaded:", len(sentences))

    if len(sentences) < TOTAL_SENTENCES:
        print(
            "\nERROR: Dataset does not contain enough sentences."
        )
        return

    # --------------------------------------------------------
    # Shuffle and split
    # --------------------------------------------------------

    random.shuffle(sentences)

    training_data = sentences[:TRAIN_SIZE]

    development_data = sentences[
        TRAIN_SIZE:TRAIN_SIZE + DEV_SIZE
    ]

    test_data = sentences[
        TRAIN_SIZE + DEV_SIZE:
        TRAIN_SIZE + DEV_SIZE + TEST_SIZE
    ]

    print("\nDataset split completed.")

    # --------------------------------------------------------
    # Training vocabulary
    # --------------------------------------------------------

    vocabulary = set()

    for sentence in training_data:
        vocabulary.update(sentence)

    vocabulary.add("<UNK>")

    training_data = replace_unknown_words(
        training_data,
        vocabulary
    )

    development_data = replace_unknown_words(
        development_data,
        vocabulary
    )

    test_data = replace_unknown_words(
        test_data,
        vocabulary
    )

    print("Vocabulary size:", len(vocabulary))

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    models = train_models(training_data)

    # --------------------------------------------------------
    # Good-Turing
    # --------------------------------------------------------

    print("\nCreating Good-Turing tables...")

    gt_tables = {
        n: create_good_turing_table(
            models[n]["ngram_counts"]
        )
        for n in range(1, 5)
    }

    # --------------------------------------------------------
    # Kneser-Ney
    # --------------------------------------------------------

    print("Preparing Kneser-Ney statistics...")

    kn = prepare_kneser_ney(models)

    # --------------------------------------------------------
    # Smoothing models
    # --------------------------------------------------------

    smoothing_models = create_smoothing_models(
        models,
        gt_tables,
        kn
    )

    # ========================================================
    # DEVELOPMENT
    # ========================================================

    print("\n" + "=" * 80)
    print("DEVELOPMENT / VALIDATION SET RESULTS")
    print("=" * 80)

    development_results = []

    for name, (n, probability_function) in smoothing_models.items():

        print(f"Running {name}...")

        ppl = calculate_perplexity(
            development_data,
            n,
            probability_function
        )

        development_results.append((name, ppl))

        print(f"  Perplexity = {ppl:.4f}")

    # ========================================================
    # TEST
    # ========================================================

    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)

    test_results = []

    for name, (n, probability_function) in smoothing_models.items():

        print(f"Running {name}...")

        ppl = calculate_perplexity(
            test_data,
            n,
            probability_function
        )

        test_results.append((name, ppl))

        print(f"  Perplexity = {ppl:.4f}")

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 90)
    print("FINAL COMPARISON")
    print("=" * 90)

    print(
        f"{'Model':<32}"
        f"{'Development PPL':>22}"
        f"{'Test PPL':>20}"
    )

    print("-" * 75)

    for i in range(len(development_results)):

        name = development_results[i][0]
        dev_ppl = development_results[i][1]
        test_ppl = test_results[i][1]

        print(
            f"{name:<32}"
            f"{dev_ppl:>22.4f}"
            f"{test_ppl:>20.4f}"
        )

    # ========================================================
    # BEST RESULTS
    # ========================================================

    best_dev = min(
        development_results,
        key=lambda x: x[1]
    )

    best_test = min(
        test_results,
        key=lambda x: x[1]
    )

    print("\n" + "=" * 70)
    print("LOWEST PERPLEXITY RESULTS")
    print("=" * 70)

    print(
        f"Development: {best_dev[0]} "
        f"-> {best_dev[1]:.4f}"
    )

    print(
        f"Test:        {best_test[0]} "
        f"-> {best_test[1]:.4f}"
    )

    print("\nLower perplexity means the model assigns")
    print("higher probability to the observed test text.")


# ============================================================
# 17. RUN
# ============================================================

if __name__ == "__main__":
    main()
