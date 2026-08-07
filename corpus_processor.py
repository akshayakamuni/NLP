from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq

from tokenizer import sentence_tokenize, word_tokenize


def process_corpus(
    dataset_name,
    output_dir,
    text_field,
    load_kwargs,
    max_documents=None
):
    """
    Process a large text corpus.

    Steps:
    1. Read paragraphs from the dataset.
    2. Split paragraphs into sentences.
    3. Tokenize each sentence into words/tokens.
    4. Join tokens using spaces.
    5. Save one tokenized sentence per line in TXT.
    6. Save tokenized sentences in Parquet.
    7. Calculate corpus statistics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_file = output_dir / f"{dataset_name}_tokenized.txt"
    parquet_file = output_dir / f"{dataset_name}_tokenized.parquet"
    stats_file = output_dir / f"{dataset_name}_statistics.json"

    print("Loading dataset...")

    from datasets import load_dataset

    dataset = load_dataset(
        **load_kwargs
    )

    print("Dataset loaded successfully.")
    print("Starting tokenization...")

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    total_sentences = 0
    total_words = 0
    total_characters = 0

    unique_tokens = set()

    # ---------------------------------------------------------
    # Parquet setup
    # ---------------------------------------------------------

    parquet_writer = None

    # ---------------------------------------------------------
    # Open TXT output file
    # ---------------------------------------------------------

    with open(
        txt_file,
        "w",
        encoding="utf-8",
        newline="\n"
    ) as txt:

        document_count = 0

        for row in dataset:

            document_count += 1

            text = row.get(text_field, "")

            if not text:
                continue

            # -------------------------------------------------
            # Step 1: Paragraph -> Sentences
            # -------------------------------------------------

            sentences = sentence_tokenize(text)

            for sentence in sentences:

                # ---------------------------------------------
                # Step 2: Sentence -> Words/Tokens
                # ---------------------------------------------

                tokens = word_tokenize(sentence)

                if not tokens:
                    continue

                # ---------------------------------------------
                # Step 3: Join tokens using spaces
                # ---------------------------------------------

                tokenized_sentence = " ".join(tokens)

                # ---------------------------------------------
                # Step 4: Write one sentence per line
                # ---------------------------------------------

                txt.write(tokenized_sentence + "\n")

                # ---------------------------------------------
                # Statistics
                # ---------------------------------------------

                total_sentences += 1

                total_words += len(tokens)

                total_characters += len(tokenized_sentence)

                unique_tokens.update(tokens)

                # ---------------------------------------------
                # Save to Parquet in batches
                # ---------------------------------------------

                if parquet_writer is None:

                    table = pa.table({
                        "tokenized_sentence": [tokenized_sentence]
                    })

                    parquet_writer = pq.ParquetWriter(
                        parquet_file,
                        table.schema,
                        compression="snappy"
                    )

                    parquet_writer.write_table(table)

                else:

                    table = pa.table({
                        "tokenized_sentence": [tokenized_sentence]
                    })

                    parquet_writer.write_table(table)

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if document_count % 10000 == 0:

                print(
                    f"Processed {document_count} documents | "
                    f"Sentences: {total_sentences} | "
                    f"Words: {total_words}"
                )

            # -------------------------------------------------
            # Stop early if max_documents is specified
            # -------------------------------------------------

            if (
                max_documents is not None
                and document_count >= max_documents
            ):
                break

    # ---------------------------------------------------------
    # Close Parquet writer
    # ---------------------------------------------------------

    if parquet_writer is not None:
        parquet_writer.close()

    # ---------------------------------------------------------
    # Calculate final statistics
    # ---------------------------------------------------------

    if total_sentences > 0:
        average_sentence_length = (
            total_words / total_sentences
        )
    else:
        average_sentence_length = 0

    if total_words > 0:
        average_word_length = (
            total_characters / total_words
        )
    else:
        average_word_length = 0

    if total_words > 0:
        ttr = (
            len(unique_tokens) / total_words
        )
    else:
        ttr = 0

    statistics = {
        "total_documents": document_count,
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_characters": total_characters,
        "average_sentence_length": average_sentence_length,
        "average_word_length": average_word_length,
        "unique_tokens": len(unique_tokens),
        "type_token_ratio": ttr
    }

    # ---------------------------------------------------------
    # Save statistics
    # ---------------------------------------------------------

    with open(
        stats_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            statistics,
            f,
            ensure_ascii=False,
            indent=4
        )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\n======================================")
    print("PROCESSING COMPLETED")
    print("======================================")

    print(f"Documents        : {document_count}")
    print(f"Sentences        : {total_sentences}")
    print(f"Words            : {total_words}")
    print(f"Characters       : {total_characters}")
    print(
        f"Average sentence : "
        f"{average_sentence_length:.4f}"
    )
    print(
        f"Average word     : "
        f"{average_word_length:.4f}"
    )
    print(f"Unique tokens    : {len(unique_tokens)}")
    print(f"TTR              : {ttr:.6f}")

    print("\nFiles created:")

    print(f"TXT     : {txt_file}")
    print(f"Parquet : {parquet_file}")
    print(f"Stats   : {stats_file}")