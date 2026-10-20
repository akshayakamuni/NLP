#!/usr/bin/env python3
"""
Hindi POS Tagger: Naive Bayes + three HMM decoding methods.

Expected input format:
    word<TAB>POS_TAG
    word POS_TAG
Blank lines separate sentences. Lines beginning with # are ignored.
If your dataset uses a different format, edit read_corpus() below.

Run:
    pip install numpy pandas matplotlib scikit-learn
    python hindi_pos_tagger.py --train train.txt --dev dev.txt --test test.txt

Outputs:
    results/per_tag_f1_dev.png
    results/per_tag_f1_test.png
    results/macro_f1_comparison.png
    results/per_tag_scores_dev.csv
    results/per_tag_scores_test.csv
    results/summary.csv
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


START = "<START>"
END = "<END>"
UNK = "<UNK>"


def read_corpus(path):
    """
    Read POS-tagged data in common formats:
      1) word<TAB>tag
      2) word tag (whitespace separated)
      3) CoNLL-U: token ID, FORM, ..., UPOS, ...
      4) word/tag (slash-separated, one token per line)
    Blank lines separate sentences. Lines beginning with # are ignored.

    For ordinary two-column data, the default assumption is WORD TAG.
    If your file uses a different layout, inspect a few lines and adapt here.
    """
    sentences, sentence = [], []

    def add_token(word, tag):
        word, tag = word.strip(), tag.strip()
        if word and tag and word != "_" and tag != "_":
            sentence.append((word, tag))

    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()

            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            if line.startswith("#"):
                continue

            # CoNLL-U rows usually begin with an integer token ID and have 10 columns.
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) >= 10 and fields[0].isdigit():
                # CoNLL-U: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
                add_token(fields[1], fields[3])
                continue

            # Slash format, e.g. राम/NNP. Use only when there is one token field.
            if len(fields) == 1 and "/" in fields[0]:
                word, tag = fields[0].rsplit("/", 1)
                add_token(word, tag)
                continue

            # Common word<TAB>tag or word tag format.
            if len(fields) >= 2:
                # For rows with extra columns but not CoNLL-U, assume first=word,
                # last=tag. For two columns, this is simply WORD TAG.
                add_token(fields[0], fields[-1])

    if sentence:
        sentences.append(sentence)

    if not sentences:
        raise ValueError(
            f"No token/tag rows found in {path}. Inspect the file format; "
            "expected WORD TAG rows, WORD/TAG rows, or CoNLL-U."
        )
    return sentences


def flatten(sentences):
    words, tags = [], []
    for sent in sentences:
        for w, t in sent:
            words.append(w)
            tags.append(t)
    return words, tags


def safe_log(p):
    return math.log(max(p, 1e-300))


class NaiveBayesPOSTagger:
    """
    Greedy Naive Bayes sequence tagger.
    Uses features: length class, current word, previous word, next word,
    and previous tag. Previous tag is the predicted tag at inference time.
    Laplace smoothing is applied separately to each feature likelihood.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.tags = []
        self.tag_counts = Counter()
        self.feature_counts = defaultdict(Counter)
        self.feature_totals = defaultdict(Counter)
        self.feature_vocab = defaultdict(set)
        self.n_tokens = 0

    @staticmethod
    def length_class(word):
        return "MORE" if len(word) > 4 else "LESS"

    @staticmethod
    def features(sentence, i, previous_tag):
        word = sentence[i]
        prev_word = sentence[i - 1] if i > 0 else "<BOS>"
        next_word = sentence[i + 1] if i + 1 < len(sentence) else "<EOS>"
        return {
            "length": NaiveBayesPOSTagger.length_class(word),
            "word": word,
            "prev_word": prev_word,
            "next_word": next_word,
            "prev_tag": previous_tag,
        }

    def fit(self, corpus):
        all_tags = set()
        for sent in corpus:
            for _, tag in sent:
                all_tags.add(tag)
        self.tags = sorted(all_tags)

        for sent in corpus:
            words = [w for w, _ in sent]
            previous_gold_tag = START
            for i, (_, tag) in enumerate(sent):
                feats = self.features(words, i, previous_gold_tag)
                self.tag_counts[tag] += 1
                self.n_tokens += 1
                for name, value in feats.items():
                    self.feature_counts[tag][(name, value)] += 1
                    self.feature_totals[tag][name] += 1
                    self.feature_vocab[name].add(value)
                previous_gold_tag = tag
        return self

    def emission_log_score(self, tag, feats):
        # P(tag) with add-alpha smoothing.
        prior = (self.tag_counts[tag] + self.alpha) / (
            self.n_tokens + self.alpha * len(self.tags)
        )
        score = safe_log(prior)
        for name, value in feats.items():
            vocab_size = max(1, len(self.feature_vocab[name]) + 1)  # + unseen bucket
            count = self.feature_counts[tag][(name, value)]
            total = self.feature_totals[tag][name]
            prob = (count + self.alpha) / (total + self.alpha * vocab_size)
            score += safe_log(prob)
        return score

    def predict_sentence(self, words):
        predicted, prev_tag = [], START
        for i in range(len(words)):
            feats = self.features(words, i, prev_tag)
            tag = max(self.tags, key=lambda t: self.emission_log_score(t, feats))
            predicted.append(tag)
            prev_tag = tag
        return predicted

    def predict(self, corpus):
        return [tag for sent in corpus for tag in self.predict_sentence([w for w, _ in sent])]


class HMMPOSTagger:
    """First-order HMM with add-alpha-smoothed transitions and emissions."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.tags = []
        self.tag_counts = Counter()
        self.transition_counts = Counter()
        self.emission_counts = Counter()
        self.outgoing_counts = Counter()
        self.vocab = set()
        self.transition_log = {}
        self.emission_log = {}

    def fit(self, corpus):
        tagset, vocab = set(), set()
        for sent in corpus:
            prev = START
            for word, tag in sent:
                tagset.add(tag)
                vocab.add(word)
                self.tag_counts[tag] += 1
                self.emission_counts[(tag, word)] += 1
                self.transition_counts[(prev, tag)] += 1
                self.outgoing_counts[prev] += 1
                prev = tag
            self.transition_counts[(prev, END)] += 1
            self.outgoing_counts[prev] += 1

        self.tags = sorted(tagset)
        self.vocab = vocab
        # Include START as a possible previous state, and END as a next state.
        previous_states = [START] + self.tags
        next_states = self.tags + [END]
        for prev in previous_states:
            denom = self.outgoing_counts[prev] + self.alpha * len(next_states)
            for nxt in next_states:
                self.transition_log[(prev, nxt)] = safe_log(
                    (self.transition_counts[(prev, nxt)] + self.alpha) / denom
                )

        emission_vocab_size = len(self.vocab) + 1  # unknown word bucket
        for tag in self.tags:
            denom = self.tag_counts[tag] + self.alpha * emission_vocab_size
            for word in self.vocab:
                self.emission_log[(tag, word)] = safe_log(
                    (self.emission_counts[(tag, word)] + self.alpha) / denom
                )
            self.emission_log[(tag, UNK)] = safe_log(self.alpha / denom)
        return self

    def emit(self, tag, word):
        return self.emission_log.get((tag, word), self.emission_log[(tag, UNK)])

    def trans(self, prev, nxt):
        return self.transition_log.get((prev, nxt), -1e6)

    def viterbi_forward(self, words):
        """Standard left-to-right Viterbi with backpointers."""
        if not words:
            return []
        dp, back = [], []
        first_scores = {
            tag: self.trans(START, tag) + self.emit(tag, words[0])
            for tag in self.tags
        }
        dp.append(first_scores)
        back.append({tag: START for tag in self.tags})

        for i in range(1, len(words)):
            scores_i, back_i = {}, {}
            for tag in self.tags:
                candidates = [
                    (dp[i - 1][prev] + self.trans(prev, tag), prev)
                    for prev in self.tags
                ]
                best_score, best_prev = max(candidates)
                scores_i[tag] = best_score + self.emit(tag, words[i])
                back_i[tag] = best_prev
            dp.append(scores_i)
            back.append(back_i)

        final_tag = max(
            self.tags, key=lambda tag: dp[-1][tag] + self.trans(tag, END)
        )
        path = [final_tag]
        for i in range(len(words) - 1, 0, -1):
            path.append(back[i][path[-1]])
        return list(reversed(path))

    def viterbi_backward(self, words):
        """
        Right-to-left dynamic programming:
        beta[i][tag] is the best log-score for positions i..end given tag at i.
        Select the best first tag, then follow right-side backpointers.
        This finds the same globally optimal sequence as standard Viterbi,
        barring ties/floating-point tie-breaking.
        """
        n = len(words)
        if n == 0:
            return []
        beta = [dict() for _ in range(n)]
        next_choice = [dict() for _ in range(n)]

        for tag in self.tags:
            beta[n - 1][tag] = self.trans(tag, END)
            next_choice[n - 1][tag] = END

        for i in range(n - 2, -1, -1):
            for tag in self.tags:
                candidates = [
                    (
                        self.trans(tag, nxt) + self.emit(nxt, words[i + 1]) + beta[i + 1][nxt],
                        nxt,
                    )
                    for nxt in self.tags
                ]
                best_score, best_next = max(candidates)
                beta[i][tag] = best_score
                next_choice[i][tag] = best_next

        first_tag = max(
            self.tags,
            key=lambda tag: self.trans(START, tag) + self.emit(tag, words[0]) + beta[0][tag],
        )
        path = [first_tag]
        for i in range(n - 1):
            path.append(next_choice[i][path[-1]])
        return path

    def forward_backward(self, words):
        """
        Forward-backward posterior decoding:
        compute P(tag_i | whole sentence) for each position and choose
        the highest-posterior tag independently at each position.
        This is not necessarily the globally most probable complete sequence.
        """
        n = len(words)
        if n == 0:
            return []
        # Forward log probabilities: log-sum over all paths ending in tag.
        forward = [dict() for _ in range(n)]
        for tag in self.tags:
            forward[0][tag] = self.trans(START, tag) + self.emit(tag, words[0])
        for i in range(1, n):
            for tag in self.tags:
                vals = [
                    forward[i - 1][prev] + self.trans(prev, tag)
                    for prev in self.tags
                ]
                forward[i][tag] = logsumexp(vals) + self.emit(tag, words[i])

        # Backward log probabilities: suffix likelihood after position i's tag.
        backward = [dict() for _ in range(n)]
        for tag in self.tags:
            backward[n - 1][tag] = self.trans(tag, END)
        for i in range(n - 2, -1, -1):
            for tag in self.tags:
                vals = [
                    self.trans(tag, nxt) + self.emit(nxt, words[i + 1]) + backward[i + 1][nxt]
                    for nxt in self.tags
                ]
                backward[i][tag] = logsumexp(vals)

        # Posterior for each tag at each position is proportional to alpha * beta.
        result = []
        for i in range(n):
            result.append(max(self.tags, key=lambda tag: forward[i][tag] + backward[i][tag]))
        return result

    def predict(self, corpus, method):
        predictions = []
        for sent in corpus:
            words = [w for w, _ in sent]
            if method == "forward":
                tags = self.viterbi_forward(words)
            elif method == "backward":
                tags = self.viterbi_backward(words)
            elif method == "forward_backward":
                tags = self.forward_backward(words)
            else:
                raise ValueError(f"Unknown HMM method: {method}")
            predictions.extend(tags)
        return predictions


def logsumexp(values):
    values = list(values)
    m = max(values)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def evaluate(gold, pred, labels):
    report = classification_report(
        gold, pred, labels=labels, output_dict=True, zero_division=0
    )
    rows = []
    for tag in labels:
        r = report.get(tag, {})
        rows.append({
            "tag": tag,
            "precision": r.get("precision", 0.0),
            "recall": r.get("recall", 0.0),
            "f1": r.get("f1-score", 0.0),
            "support": r.get("support", 0),
        })
    macro = f1_score(gold, pred, labels=labels, average="macro", zero_division=0)
    weighted = f1_score(gold, pred, labels=labels, average="weighted", zero_division=0)
    return pd.DataFrame(rows), {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": macro,
        "weighted_f1": weighted,
    }


def plot_per_tag(results, split, out_dir):
    """results maps model name to a per-tag dataframe."""
    tags = sorted(set().union(*(set(df["tag"]) for df in results.values())))
    x = np.arange(len(tags))
    width = 0.8 / len(results)
    fig_width = max(12, len(tags) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    for j, (model, df) in enumerate(results.items()):
        lookup = dict(zip(df["tag"], df["f1"]))
        vals = [lookup.get(tag, 0.0) for tag in tags]
        ax.bar(x + (j - (len(results) - 1) / 2) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=60, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_xlabel("POS tag")
    ax.set_title(f"Per-tag F1 comparison ({split} split)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"per_tag_f1_{split}.png", dpi=180)
    plt.close(fig)


def plot_macro(summary, out_dir):
    for split in sorted(summary["split"].unique()):
        sub = summary[summary["split"] == split]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(sub["model"], sub["macro_f1"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Macro-F1 comparison ({split} split)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"macro_f1_comparison_{split}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="hindi-train-pos.txt", help="Path to training corpus")
    parser.add_argument("--dev", default="hindi-dev-pos.txt", help="Path to development corpus")
    parser.add_argument("--test", default="hindi-test-pos.txt", help="Path to test corpus")
    parser.add_argument("--alpha-nb", type=float, default=1.0, help="NB Laplace smoothing")
    parser.add_argument("--alpha-hmm", type=float, default=0.1, help="HMM smoothing")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = read_corpus(args.train)
    dev = read_corpus(args.dev)
    test = read_corpus(args.test)

    print(f"Sentences: train={len(train)}, dev={len(dev)}, test={len(test)}")
    print(f"Tokens: train={sum(map(len, train))}, dev={sum(map(len, dev))}, test={sum(map(len, test))}")

    nb = NaiveBayesPOSTagger(alpha=args.alpha_nb).fit(train)
    hmm = HMMPOSTagger(alpha=args.alpha_hmm).fit(train)

    model_predictions = {
        "Naive Bayes": {
            "dev": nb.predict(dev),
            "test": nb.predict(test),
        },
        "HMM Forward Viterbi": {
            "dev": hmm.predict(dev, "forward"),
            "test": hmm.predict(test, "forward"),
        },
        "HMM Backward Viterbi": {
            "dev": hmm.predict(dev, "backward"),
            "test": hmm.predict(test, "backward"),
        },
        "HMM Forward-Backward": {
            "dev": hmm.predict(dev, "forward_backward"),
            "test": hmm.predict(test, "forward_backward"),
        },
    }

    summary_rows = []
    labels = sorted({tag for sent in train for _, tag in sent})
    for split, corpus in [("dev", dev), ("test", test)]:
        gold = [tag for sent in corpus for _, tag in sent]
        per_model = {}
        for model_name, split_preds in model_predictions.items():
            pred = split_preds[split]
            per_tag, metrics = evaluate(gold, pred, labels)
            per_model[model_name] = per_tag
            per_tag.to_csv(out_dir / f"{model_name.lower().replace(' ', '_').replace('-', '_')}_per_tag_{split}.csv", index=False)
            summary_rows.append({
                "split": split,
                "model": model_name,
                **metrics,
            })
            print(f"\n{split.upper()} — {model_name}")
            print(f"Accuracy={metrics['accuracy']:.4f} | Macro-F1={metrics['macro_f1']:.4f} | Weighted-F1={metrics['weighted_f1']:.4f}")
        combined = []
        for model_name, df in per_model.items():
            tmp = df.copy()
            tmp["model"] = model_name
            combined.append(tmp)
        pd.concat(combined, ignore_index=True).to_csv(
            out_dir / f"per_tag_scores_{split}.csv", index=False
        )
        plot_per_tag(per_model, split, out_dir)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot_macro(summary, out_dir)

    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved metrics and charts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
