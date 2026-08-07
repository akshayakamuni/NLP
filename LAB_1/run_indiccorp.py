from pathlib import Path

from corpus_processor import process_corpus


OUTPUT_DIR = Path(__file__).parent / "output" / "indiccorp"

MAX_DOCUMENTS = 1000000


if __name__ == "__main__":

    process_corpus(
        dataset_name="indiccorp_te",

        output_dir=OUTPUT_DIR,

        text_field="text",

        load_kwargs={
            "path": "ai4bharat/IndicCorpV2",
            "name": "indiccorp_v2",
            "split": "tel_Telu",
            "streaming": True
        },

        max_documents=MAX_DOCUMENTS
    )