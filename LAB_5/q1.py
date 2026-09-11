# import re
# from collections import Counter
# from math import log2


# # ============================================================
# # 1. TOKENIZATION
# # ============================================================

# def tokenize(text):
#     text = text.lower()
#     return re.findall(r"\b\w+\b|[.!?,;]", text)


# # ============================================================
# # 2. N-GRAM MODEL
# # ============================================================

# class NGramModel:

#     def __init__(self, n, k=0.3):
#         self.n = n
#         self.k = k

#         self.ngram_counts = Counter()
#         self.context_counts = Counter()
#         self.vocab = set()

#     def train(self, text):

#         tokens = tokenize(text)

#         self.vocab = set(tokens)

#         # Add sentence boundary tokens
#         tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]

#         # Count n-grams
#         for i in range(len(tokens) - self.n + 1):

#             ngram = tuple(tokens[i:i + self.n])
#             context = tuple(tokens[i:i + self.n - 1])

#             self.ngram_counts[ngram] += 1
#             self.context_counts[context] += 1

#     # ========================================================
#     # ADD-K PROBABILITY
#     # ========================================================

#     def probability(self, word, context):

#         # Unigram
#         if self.n == 1:
#             count_ngram = self.ngram_counts[(word,)]
#             denominator_count = sum(self.ngram_counts.values())

#         else:
#             context = tuple(context[-(self.n - 1):])

#             count_ngram = self.ngram_counts[
#                 context + (word,)
#             ]

#             denominator_count = self.context_counts[context]

#         V = len(self.vocab)

#         probability = (
#             count_ngram + self.k
#         ) / (
#             denominator_count + self.k * V
#         )

#         return probability

#     # ========================================================
#     # CALCULATE SENTENCE PROBABILITY
#     # ========================================================

#     def sentence_probability(self, sentence):

#         tokens = tokenize(sentence)

#         tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]

#         probability = 1.0

#         for i in range(self.n - 1, len(tokens)):

#             word = tokens[i]

#             if self.n == 1:
#                 context = ()
#             else:
#                 context = tokens[
#                     i - self.n + 1:i
#                 ]

#             probability *= self.probability(
#                 word,
#                 context
#             )

#         return probability

#     # ========================================================
#     # PERPLEXITY
#     # ========================================================

#     def perplexity(self, text):

#         tokens = tokenize(text)

#         tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]

#         log_probability = 0
#         N = 0

#         for i in range(self.n - 1, len(tokens)):

#             word = tokens[i]

#             if self.n == 1:
#                 context = ()
#             else:
#                 context = tokens[
#                     i - self.n + 1:i
#                 ]

#             p = self.probability(word, context)

#             log_probability += log2(p)
#             N += 1

#         return 2 ** (-log_probability / N)


# # ============================================================
# # 3. TRAINING DATA
# # ============================================================

# train_text = """
# I love natural language processing.
# I love machine learning.
# Machine learning is interesting.
# Natural language processing is interesting.
# I study machine learning.
# I study natural language processing.
# """


# # ============================================================
# # 4. VALIDATION / TEST DATA
# # ============================================================

# validation_text = """
# I love machine learning.
# Natural language processing is interesting.
# I study machine learning.
# """


# # ============================================================
# # 5. TRAIN ALL 4 MODELS
# # ============================================================

# k = 0.3

# models = {}

# for n in range(1, 5):

#     model = NGramModel(n=n, k=k)

#     model.train(train_text)

#     models[n] = model


# # ============================================================
# # 6. TEST SENTENCE PROBABILITY
# # ============================================================

# sentence = "I love machine learning."

# print("Sentence:", sentence)
# print()

# for n, model in models.items():

#     p = model.sentence_probability(sentence)

#     print(
#         f"{n}-gram probability: {p:.10f}"
#     )


# # ============================================================
# # 7. VALIDATION PERPLEXITY
# # ============================================================

# print("\nValidation Perplexity")
# print("---------------------")

# for n, model in models.items():

#     pp = model.perplexity(validation_text)

#     print(
#         f"{n}-gram: {pp:.4f}"
#     )

from collections import Counter
import math
import random
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FILE_NAME = (
    Path(__file__).resolve().parent.parent
    / "LAB_4" / "indiccorp_te_tokenized.txt"
)

TOTAL_SENTENCES = 100000
TRAIN_SIZE = 98000
DEV_SIZE = 1000
TEST_SIZE = 1000
ADD_K = 0.3

random.seed(42)


# ============================================================
# 1. READ TOKENIZED SENTENCES
# ============================================================

def load_sentences(filename, limit=100000):
    sentences = []

    if str(filename).lower().endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(filename, columns=["tokenized_sentence"])

        for sentence in table.column("tokenized_sentence").to_pylist():
            if sentence and sentence.strip():
                sentences.append(sentence.strip().split())

            if len(sentences) == limit:
                break

        return sentences

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            tokens = line.split()

            if len(tokens) > 0:
                sentences.append(tokens)

            if len(sentences) == limit:
                break

    return sentences


# ============================================================
# 2. REPLACE WORDS NOT IN TRAINING VOCABULARY WITH <UNK>
# ============================================================

def replace_unknown_words(sentences, vocabulary):
    result = []

    for sentence in sentences:
        new_sentence = []
        for word in sentence:
            if word in vocabulary:
                new_sentence.append(word)
            else:
                new_sentence.append("<UNK>")
        result.append(new_sentence)

    return result


# ============================================================
# 3. BUILD N-GRAM MODEL
# ============================================================

def build_ngram_model(sentences, n):
    ngram_counts = Counter()
    context_counts = Counter()
    vocabulary = set()

    for sentence in sentences:
        if n == 1:
            tokens = sentence + ["</s>"]
        else:
            tokens = ["<s>"] * (n - 1) + sentence + ["</s>"]

        for word in tokens:
            vocabulary.add(word)

        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngram_counts[ngram] += 1

            if n > 1:
                context = tuple(tokens[i:i + n - 1])
                context_counts[context] += 1

    return ngram_counts, context_counts, vocabulary


# ============================================================
# 4. ADD-K SMOOTHING PROBABILITY
# ============================================================

def add_k_probability(ngram, ngram_counts, context_counts, vocabulary, total_tokens, k=ADD_K):
    V = len(vocabulary)
    count_ngram = ngram_counts.get(ngram, 0)

    if len(ngram) == 1:
        return (count_ngram + k) / (total_tokens + k * V)

    context = ngram[:-1]
    count_context = context_counts.get(context, 0)
    return (count_ngram + k) / (count_context + k * V)


# ============================================================
# 5. CALCULATE SENTENCE LOG PROBABILITY
# ============================================================

def sentence_log_probability(sentence, n, ngram_counts, context_counts, vocabulary, total_tokens, k=ADD_K):
    if n == 1:
        tokens = sentence + ["</s>"]
    else:
        tokens = ["<s>"] * (n - 1) + sentence + ["</s>"]

    log_probability = 0.0

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        probability = add_k_probability(
            ngram,
            ngram_counts,
            context_counts,
            vocabulary,
            total_tokens,
            k=k
        )
        log_probability += math.log(probability)

    return log_probability


# ============================================================
# 6. CALCULATE PERPLEXITY
# ============================================================

def calculate_perplexity(sentences, n, ngram_counts, context_counts, vocabulary, total_tokens, k=ADD_K):
    total_log_probability = 0.0
    total_ngrams = 0

    for sentence in sentences:
        if n == 1:
            tokens = sentence + ["</s>"]
        else:
            tokens = ["<s>"] * (n - 1) + sentence + ["</s>"]

        log_probability = sentence_log_probability(
            sentence,
            n,
            ngram_counts,
            context_counts,
            vocabulary,
            total_tokens,
            k=k
        )

        total_log_probability += log_probability
        total_ngrams += len(tokens) - n + 1

    return math.exp(-total_log_probability / total_ngrams)


# ============================================================
# 7. TRAIN ONE MODEL
# ============================================================

def train_model(training_data, n):
    print("\n" + "=" * 60)

    if n == 1:
        model_label = "UNIGRAM"
    elif n == 2:
        model_label = "BIGRAM"
    elif n == 3:
        model_label = "TRIGRAM"
    else:
        model_label = "QUADGRAM"

    print(f"TRAINING {model_label} MODEL WITH ADD-K (k={ADD_K})")
    print("=" * 60)

    ngram_counts, context_counts, vocabulary = build_ngram_model(training_data, n)
    total_tokens = sum(ngram_counts.values())

    print("Vocabulary size:", len(vocabulary))
    print("N-gram count   :", len(ngram_counts))

    return {
        "ngram_counts": ngram_counts,
        "context_counts": context_counts,
        "vocabulary": vocabulary,
        "total_tokens": total_tokens,
    }


# ============================================================
# 8. DISPLAY SAMPLE PROBABILITIES
# ============================================================

def show_sample_probabilities(models):
    print("\n")
    print("=" * 70)
    print("SAMPLE ADD-K PROBABILITIES")
    print("=" * 70)

    model = models[1]
    word = next(word for word in model["vocabulary"] if word not in {"<s>", "</s>", "<UNK>"})
    unigram = (word,)
    probability = add_k_probability(
        unigram,
        model["ngram_counts"],
        model["context_counts"],
        model["vocabulary"],
        model["total_tokens"],
        k=ADD_K,
    )
    print("\nUnigram:")
    print("Word:", word)
    print("P(" + word + ") =", probability)

    model = models[2]
    bigram = next(iter(model["ngram_counts"]))
    probability = add_k_probability(
        bigram,
        model["ngram_counts"],
        model["context_counts"],
        model["vocabulary"],
        model["total_tokens"],
        k=ADD_K,
    )
    print("\nBigram:")
    print("Bigram:", bigram)
    print("Probability =", probability)

    model = models[3]
    trigram = next(iter(model["ngram_counts"]))
    probability = add_k_probability(
        trigram,
        model["ngram_counts"],
        model["context_counts"],
        model["vocabulary"],
        model["total_tokens"],
        k=ADD_K,
    )
    print("\nTrigram:")
    print("Trigram:", trigram)
    print("Probability =", probability)

    model = models[4]
    quadgram = next(iter(model["ngram_counts"]))
    probability = add_k_probability(
        quadgram,
        model["ngram_counts"],
        model["context_counts"],
        model["vocabulary"],
        model["total_tokens"],
        k=ADD_K,
    )
    print("\nQuadgram:")
    print("Quadgram:", quadgram)
    print("Probability =", probability)


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():
    print("=" * 70)
    print("HINDI N-GRAM LANGUAGE MODEL")
    print("ADD-K SMOOTHING")
    print(f"K = {ADD_K}")
    print("=" * 70)

    print("\nReading dataset...")
    sentences = load_sentences(FILE_NAME, TOTAL_SENTENCES)
    print("Sentences loaded:", len(sentences))

    if len(sentences) < TOTAL_SENTENCES:
        print("\nERROR: Dataset contains fewer than 100,000 sentences.")
        return

    random.shuffle(sentences)

    training_data = sentences[:TRAIN_SIZE]
    development_data = sentences[TRAIN_SIZE:TRAIN_SIZE + DEV_SIZE]
    test_data = sentences[TRAIN_SIZE + DEV_SIZE:TOTAL_SENTENCES]

    print("\n" + "=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)
    print("Training sentences    :", len(training_data))
    print("Development sentences :", len(development_data))
    print("Test sentences        :", len(test_data))

    training_vocabulary = set()
    for sentence in training_data:
        for word in sentence:
            training_vocabulary.add(word)
    training_vocabulary.add("<UNK>")

    development_data = replace_unknown_words(development_data, training_vocabulary)
    test_data = replace_unknown_words(test_data, training_vocabulary)
    training_data = replace_unknown_words(training_data, training_vocabulary)

    models = {}
    for n in range(1, 5):
        models[n] = train_model(training_data, n)

    show_sample_probabilities(models)

    print("\n")
    print("=" * 70)
    print("PERPLEXITY RESULTS (ADD-K, K=0.3)")
    print("=" * 70)

    print("\n{:<15} {:>20} {:>20}".format("Model", "Development", "Test"))
    print("-" * 60)

    for n in range(1, 5):
        model = models[n]

        dev_perplexity = calculate_perplexity(
            development_data,
            n,
            model["ngram_counts"],
            model["context_counts"],
            model["vocabulary"],
            model["total_tokens"],
            k=ADD_K,
        )

        test_perplexity = calculate_perplexity(
            test_data,
            n,
            model["ngram_counts"],
            model["context_counts"],
            model["vocabulary"],
            model["total_tokens"],
            k=ADD_K,
        )

        if n == 1:
            model_name = "Unigram"
        elif n == 2:
            model_name = "Bigram"
        elif n == 3:
            model_name = "Trigram"
        else:
            model_name = "Quadgram"

        print(
            "{:<15} {:>20.4f} {:>20.4f}".format(
                model_name,
                dev_perplexity,
                test_perplexity,
            )
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
