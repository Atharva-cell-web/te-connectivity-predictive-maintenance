import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

CSV_FILE = Path("processed/features/rolling_training_labeled.csv")
PARQUET_FILE = Path("processed/features/rolling_training_labeled.parquet")

CHUNK_SIZE = 500_000

print("Streaming CSV → Parquet (low-memory mode)")

writer = None
total_rows = 0

for i, chunk in enumerate(pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE)):
    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(PARQUET_FILE, table.schema)

    writer.write_table(table)
    total_rows += len(chunk)

    if (i + 1) % 10 == 0:
        print(f"Written {total_rows:,} rows")

writer.close()

print("Parquet conversion completed successfully")
print("Total rows written:", total_rows)
print("Saved to:", PARQUET_FILE)
