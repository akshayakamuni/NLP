"""
Hindi POS Tagging Assignment: Naive Bayes, HMM, MEMM, and CRF
----------------------------------------------------------------
Expected files in the same folder:
    hindi-train-pos.txt
    hindi-dev-pos.txt
    hindi-test-pos.txt

Supported corpus formats:
  1) Two columns per token: WORD TAG
  2) CoNLL-U-like rows: ID WORD ... UPOS ... (word column 2, tag column 4)
  3) Slash-separated tokens: word/TAG word/TAG ... (blank line = sentence)
  4) One token per line as WORD/TAG or WORD TAG

Blank lines delimit sentences. Lines beginning with # are ignored.

Install:
    python -m pip install nltk scikit-learn sklearn-crfsuite pandas matplotlib

Run:
    python pos_tagging_assignment.py

Notes:
- MEMM uses NLTK's Maximum Entropy classifier with greedy left-to-right decoding.
  The feature design follows the requested assignment features and the
  arne-cl/nltk-maxent-pos-tagger maximum-entropy POS-tagger approach.
- CRF uses sklearn-crfsuite.
- NB and HMM are self-contained baselines so the comparative charts can be
  generated from one run. If your Assignment 7 implementations differ, replace
  these baseline classes with your original implementations for an exact
  apples-to-apples comparison.
"""

from pathlib import Path
from collections import Counter, defaultdict
import math
import re
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)

try:
    import nltk
    from nltk.classify import MaxentClassifier
except ImportError as exc:
    raise SystemExit(
        "NLTK is missing. Install it with: python -m pip install nltk"
    ) from exc

try:
    import sklearn_crfsuite
except ImportError as exc:
    raise SystemExit(
        "sklearn-crfsuite is missing. Install it with: "
        "python -m pip install sklearn-crfsuite"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "hindi-train-pos.txt"
DEV_FILE = BASE_DIR / "hindi-dev-pos.txt"
TEST_FILE = BASE_DIR / "hindi-test-pos.txt"
OUT_DIR = BASE_DIR / "pos_tagging_results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Dataset reader
# ---------------------------------------------------------------------------

def parse_token_line(line):
    """Return (word, tag) for a single token row, or None if unrecognized."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # CoNLL-U: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
    cols = line.split()
    if len(cols) >= 4 and re.fullmatch(r"\d+", cols[0]):
        word, tag = cols[1], cols[3]
        if word and tag and tag != "_":
            return word, tag

    # Tabular / whitespace-delimited WORD TAG format
    if len(cols) == 2:
        word, tag = cols
        # Ignore a possible CoNLL header or malformed row
        if word not in {"#", "-DOCSTART-"} and tag != "_":
            return word, tag

    # Some CoNLL-like formats put tag in final column
    if len(cols) >= 3 and not re.fullmatch(r"\d+", cols[0]):
        # Prefer a clear word/tag pair only; otherwise final-column fallback.
        if cols[-1] != "_" and cols[0] not in {"-DOCSTART-"}:
            return cols[0], cols[-1]

    # Slash-separated token such as राम/NNP
    if "/" in line and len(cols) == 1:
        # Split at the last slash to allow slash characters in the word.
        word, tag = line.rsplit("/", 1)
        if word and tag:
            return word, tag

    return None


def read_corpus(path):
    """Read sentences as list[list[(word, tag)]]."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name} in {BASE_DIR}\n"
            "Place the train/dev/test files beside this Python script."
        )

    sentences, sentence = [], []
    with path.open("r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            if line.startswith("#"):
                continue

            # A line containing several slash-separated tokens.
            cols = line.split()
            if len(cols) > 2 and all("/" in token for token in cols):
                parsed = []
                for token in cols:
                    pair = parse_token_line(token)
                    if pair:
                        parsed.append(pair)
                if parsed:
                    sentence.extend(parsed)
                continue

            pair = parse_token_line(line)
            if pair:
                sentence.append(pair)
            else:
                # A token row we cannot parse should be reported, not silently
                # treated as a word or tag.
                warnings.warn(f"Skipped unrecognized line in {path.name}: {line}")

    if sentence:
        sentences.append(sentence)

    if not sentences:
        raise ValueError(
            f"No tagged sentences could be read from {path.name}. "
            "Check the file format and delimiter."
        )
    return sentences


# ---------------------------------------------------------------------------
# 2. Common feature extraction (used by MEMM and CRF)
# ---------------------------------------------------------------------------

def safe_affix(word, length, mode):
    """Return an affix, or literal NULL if the word is shorter than length."""
    if len(word) < length:
        return "NULL"
    return word[:length] if mode == "prefix" else word[-length:]


def word_features(words, i, previous_tag=None):
    """Assignment features: prefix/suffix 1..4, context words, binary length."""
    word = words[i]
    prev_word = words[i - 1] if i > 0 else "<START>"
    next_word = words[i + 1] if i < len(words) - 1 else "<END>"

    features = {
        "previous_word": prev_word,
        "current_word": word,
        "next_word": next_word,
        "length_feature": "LESS" if len(word) <= 4 else "MORE",
    }

    for n in range(1, 5):
        features[f"prefix_{n}"] = safe_affix(word, n, "prefix")
        features[f"suffix_{n}"] = safe_affix(word, n, "suffix")

    # MEMM uses the preceding tag as a transition/context feature.
    if previous_tag is not None:
        features["previous_tag"] = previous_tag

    return features


def sent_words(sentence):
    return [word for word, _ in sentence]


def sent_tags(sentence):
    return [tag for _, tag in sentence]


# ---------------------------------------------------------------------------
# 3. MEMM: Maximum Entropy Markov Model
# ---------------------------------------------------------------------------

class MEMMPOSTagger:
    """
    A first-order MEMM:
      P(tag_i | words, tag_{i-1})
    Training uses the gold previous tag; inference uses the previously
    predicted tag (greedy left-to-right decoding).
    """

    def __init__(self, max_iter=20, min_feature_freq=1):
        self.max_iter = max_iter
        self.min_feature_freq = min_feature_freq
        self.classifier = None
        self.tags = []

    def train(self, sentences):
        training_instances = []
        tag_counts = Counter()

        for sentence in sentences:
            words = sent_words(sentence)
            tags = sent_tags(sentence)
            previous_tag = "<START>"
            for i, gold_tag in enumerate(tags):
                features = word_features(words, i, previous_tag)
                training_instances.append((features, gold_tag))
                tag_counts[gold_tag] += 1
                previous_tag = gold_tag

        self.tags = sorted(tag_counts)
        # NLTK's MaxentClassifier is the core maximum-entropy classifier used
        # by the referenced NLTK-based repository. IIS is portable and does
        # not require the older external MEGAM binary.
        self.classifier = MaxentClassifier.train(
            training_instances,
            algorithm="IIS",
            trace=0,
            max_iter=self.max_iter,
            min_lldelta=0.01,
        )
        return self

    def predict_sentence(self, words):
        if self.classifier is None:
            raise RuntimeError("Train MEMM before prediction.")
        predictions = []
        previous_tag = "<START>"
        for i in range(len(words)):
            features = word_features(words, i, previous_tag)
            predicted_tag = self.classifier.classify(features)
            predictions.append(predicted_tag)
            previous_tag = predicted_tag
        return predictions

    def predict(self, sentences):
        return [self.predict_sentence(sent_words(s)) for s in sentences]


# ---------------------------------------------------------------------------
# 4. CRF: sklearn-crfsuite
# ---------------------------------------------------------------------------

class CRFPOSTagger:
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )

    def train(self, sentences):
        X = [
            [word_features(sent_words(sentence), i) for i in range(len(sentence))]
            for sentence in sentences
        ]
        y = [sent_tags(sentence) for sentence in sentences]
        self.model.fit(X, y)
        return self

    def predict(self, sentences):
        X = [
            [word_features(sent_words(sentence), i) for i in range(len(sentence))]
            for sentence in sentences
        ]
        return self.model.predict(X)


# ---------------------------------------------------------------------------
# 5. Naive Bayes baseline (self-contained)
# ---------------------------------------------------------------------------

class NaiveBayesPOSTagger:
    """Multinomial feature likelihood model with add-alpha smoothing."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.tag_counts = Counter()
        self.feature_counts = defaultdict(Counter)
        self.feature_vocab = defaultdict(set)
        self.tags = []

    def _features(self, words, i, previous_tag):
        # Keeps the Assignment 7 baseline style: word/context/length/previous tag.
        w = words[i]
        return {
            "word": w,
            "previous_word": words[i - 1] if i > 0 else "<START>",
            "next_word": words[i + 1] if i + 1 < len(words) else "<END>",
            "length_feature": "LESS" if len(w) <= 4 else "MORE",
            "previous_tag": previous_tag,
        }

    def train(self, sentences):
        for sentence in sentences:
            words, tags = sent_words(sentence), sent_tags(sentence)
            prev = "<START>"
            for i, tag in enumerate(tags):
                self.tag_counts[tag] += 1
                for fname, fvalue in self._features(words, i, prev).items():
                    key = f"{fname}={fvalue}"
                    self.feature_counts[tag][key] += 1
                    self.feature_vocab[fname].add(fvalue)
                prev = tag
        self.tags = sorted(self.tag_counts)
        return self

    def predict_sentence(self, words):
        output, prev = [], "<START>"
        total_tokens = sum(self.tag_counts.values())
        for i in range(len(words)):
            feats = self._features(words, i, prev)
            best_tag, best_score = None, -float("inf")
            for tag in self.tags:
                score = math.log((self.tag_counts[tag] + self.alpha) /
                                 (total_tokens + self.alpha * len(self.tags)))
                for fname, fvalue in feats.items():
                    key = f"{fname}={fvalue}"
                    vocab_size = max(1, len(self.feature_vocab[fname]))
                    numerator = self.feature_counts[tag][key] + self.alpha
                    denominator = self.tag_counts[tag] + self.alpha * vocab_size
                    score += math.log(numerator / denominator)
                if score > best_score:
                    best_tag, best_score = tag, score
            output.append(best_tag)
            prev = best_tag
        return output

    def predict(self, sentences):
        return [self.predict_sentence(sent_words(s)) for s in sentences]


# ---------------------------------------------------------------------------
# 6. HMM baseline with Viterbi decoding
# ---------------------------------------------------------------------------

class HMMPOSTagger:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.tags = []
        self.tag_counts = Counter()
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.vocab = set()
        self.total_tokens = 0

    def train(self, sentences):
        for sentence in sentences:
            words, tags = sent_words(sentence), sent_tags(sentence)
            previous = "<START>"
            for word, tag in zip(words, tags):
                self.tag_counts[tag] += 1
                self.emission_counts[tag][word] += 1
                self.transition_counts[previous][tag] += 1
                self.vocab.add(word)
                self.total_tokens += 1
                previous = tag
            self.transition_counts[previous]["<END>"] += 1

        self.tags = sorted(self.tag_counts)
        self.vocab.add("<UNK>")
        return self

    def _log_transition(self, previous, tag):
        possible = len(self.tags) + 1  # include END
        count = self.transition_counts[previous][tag]
        total = sum(self.transition_counts[previous].values())
        return math.log((count + self.alpha) /
                        (total + self.alpha * possible))

    def _log_emission(self, tag, word):
        word = word if word in self.vocab else "<UNK>"
        # UNK gets smoothing even if not seen explicitly.
        count = self.emission_counts[tag][word]
        vocab_size = max(1, len(self.vocab))
        return math.log((count + self.alpha) /
                        (self.tag_counts[tag] + self.alpha * vocab_size))

    def predict_sentence(self, words):
        if not words:
            return []
        # Viterbi in log space
        dp, back = [{}], [{}]
        for tag in self.tags:
            dp[0][tag] = (
                self._log_transition("<START>", tag)
                + self._log_emission(tag, words[0])
            )
            back[0][tag] = None

        for i in range(1, len(words)):
            dp.append({})
            back.append({})
            for tag in self.tags:
                best_prev, best_score = None, -float("inf")
                emit = self._log_emission(tag, words[i])
                for prev in self.tags:
                    score = dp[i - 1][prev] + self._log_transition(prev, tag) + emit
                    if score > best_score:
                        best_prev, best_score = prev, score
                dp[i][tag] = best_score
                back[i][tag] = best_prev

        last_tag = max(
            self.tags,
            key=lambda tag: dp[-1][tag] + self._log_transition(tag, "<END>")
        )
        sequence = [last_tag]
        for i in range(len(words) - 1, 0, -1):
            sequence.append(back[i][sequence[-1]])
        sequence.reverse()
        return sequence

    def predict(self, sentences):
        return [self.predict_sentence(sent_words(s)) for s in sentences]


# ---------------------------------------------------------------------------
# 7. Evaluation, CSVs, and charts
# ---------------------------------------------------------------------------

def flatten_gold(sentences):
    return [tag for sentence in sentences for _, tag in sentence]


def flatten_predictions(predictions):
    return [tag for sentence in predictions for tag in sentence]


def evaluate_model(model_name, gold_sentences, predictions, split_name):
    y_true = flatten_gold(gold_sentences)
    y_pred = flatten_predictions(predictions)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"{model_name} produced {len(y_pred)} tags but gold has {len(y_true)} "
            f"tokens on {split_name}. Check corpus parsing and sentence boundaries."
        )

    labels = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    report = pd.DataFrame({
        "tag": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    report.insert(0, "model", model_name)
    report.insert(1, "split", split_name)
    report["accuracy"] = accuracy
    report.to_csv(OUT_DIR / f"{model_name.lower()}_{split_name}_per_tag.csv", index=False)

    print(f"\n{'=' * 72}\n{model_name} - {split_name.upper()}")
    print(f"Token accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4))

    macro_f1 = float(report["f1"].mean())
    weighted_f1 = float(
        (report["f1"] * report["support"]).sum() / max(1, report["support"].sum())
    )
    summary = {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "tokens": len(y_true),
    }
    return summary, report


def save_comparison_charts(summary_df, per_tag_df, split_name):
    subset = summary_df[summary_df["split"] == split_name]
    if not subset.empty:
        ax = subset.set_index("model")[["accuracy", "macro_f1", "weighted_f1"]].plot(
            kind="bar", figsize=(10, 6), ylim=(0, 1), rot=0
        )
        ax.set_title(f"Overall Model Comparison - {split_name.upper()}")
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"overall_comparison_{split_name}.png", dpi=200)
        plt.close()

    subset = per_tag_df[per_tag_df["split"] == split_name]
    if not subset.empty:
        pivot = subset.pivot_table(index="tag", columns="model", values="f1", aggfunc="mean")
        ax = pivot.plot(kind="bar", figsize=(max(12, len(pivot) * 0.65), 7), ylim=(0, 1))
        ax.set_title(f"Per-Tag F1 Comparison - {split_name.upper()}")
        ax.set_ylabel("F1-score")
        ax.set_xlabel("POS tag")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"per_tag_f1_comparison_{split_name}.png", dpi=200)
        plt.close()

    # Separate per-tag metric charts, if the report requires each tag metric.
    for metric in ("precision", "recall", "f1"):
        if not subset.empty:
            pivot = subset.pivot_table(index="tag", columns="model", values=metric, aggfunc="mean")
            ax = pivot.plot(kind="bar", figsize=(max(12, len(pivot) * 0.65), 7), ylim=(0, 1))
            ax.set_title(f"Per-Tag {metric.title()} Comparison - {split_name.upper()}")
            ax.set_ylabel(metric.title())
            ax.set_xlabel("POS tag")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"per_tag_{metric}_{split_name}.png", dpi=200)
            plt.close()


def main():
    print("Loading Assignment 7 Hindi POS dataset...")
    train_sentences = read_corpus(TRAIN_FILE)
    dev_sentences = read_corpus(DEV_FILE)
    test_sentences = read_corpus(TEST_FILE)

    for name, data in [
        ("TRAIN", train_sentences), ("DEV", dev_sentences), ("TEST", test_sentences)
    ]:
        tokens = sum(len(s) for s in data)
        print(f"{name}: {len(data)} sentences, {tokens} tokens")

    # Model list. All models are trained ONLY on train.
    models = [
        ("NaiveBayes", NaiveBayesPOSTagger(alpha=1.0)),
        ("HMM", HMMPOSTagger(alpha=0.1)),
        ("MEMM", MEMMPOSTagger(max_iter=20)),
        ("CRF", CRFPOSTagger()),
    ]

    summaries, per_tag_reports = [], []
    for name, model in models:
        print(f"\nTraining {name}...")
        model.train(train_sentences)

        for split_name, gold in (("dev", dev_sentences), ("test", test_sentences)):
            predictions = model.predict(gold)
            summary, report = evaluate_model(name, gold, predictions, split_name)
            summaries.append(summary)
            per_tag_reports.append(report)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    per_tag_df = pd.concat(per_tag_reports, ignore_index=True)
    per_tag_df.to_csv(OUT_DIR / "all_models_per_tag_metrics.csv", index=False)

    for split_name in ("dev", "test"):
        save_comparison_charts(summary_df, per_tag_df, split_name)

    print("\n\nFINAL SUMMARY")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nAll reports and plots saved in:\n{OUT_DIR}")
    print("\nImportant: Use the same exact NB/HMM code from Assignment 7 if your "
          "professor expects the original baselines. This script includes "
          "self-contained baselines only so all four model charts can be produced.")


if __name__ == "__main__":
    main()
