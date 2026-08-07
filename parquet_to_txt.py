import pyarrow.parquet as pq

input_file = "output/indiccorp/indiccorp_te_tokenized.parquet"
output_file = "output/indiccorp/indiccorp_te_tokenized.txt"

pf = pq.ParquetFile(input_file)

with open(output_file, "w", encoding="utf-8") as f:

    for batch in pf.iter_batches(batch_size=10000):
        data = batch.to_pydict()

        for sentence in data["tokenized_sentence"]:
            f.write(sentence + "\n")

print("TXT file created successfully!")