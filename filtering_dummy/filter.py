import pandas as pd


# Columns to keep
#COLUMNS_TO_KEEP = ['id', 'target_id', 'target_short_name', 'target_nation', 'applicant']
COLUMNS_TO_KEEP = ['id', 'acquiror_id', 'acquiror_short_name', 'acquiror_nation', 'applicant']


def filter_columns(input_file: str, output_file: str, chunk_size: int = 100000) -> None:
    """
    Keep only specified columns from a large CSV file and save to a single file.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
        chunk_size: Number of rows to process at a time (default: 100000)
    """
    # Read the first chunk to verify columns exist
    first_chunk = pd.read_csv(input_file, nrows=0)
    existing_columns = first_chunk.columns.tolist()
    
    # Check which columns exist
    valid_columns = [col for col in COLUMNS_TO_KEEP if col in existing_columns]
    missing_columns = [col for col in COLUMNS_TO_KEEP if col not in existing_columns]
    
    if missing_columns:
        print(f"Warning: These columns not found: {missing_columns}")
    
    print(f"Keeping columns: {valid_columns}")
    print("-" * 50)
    
    # Process the file in chunks
    chunks_processed = 0
    total_rows = 0
    
    for chunk in pd.read_csv(input_file, usecols=valid_columns, chunksize=chunk_size):
        # Write header only for the first chunk
        if chunks_processed == 0:
            chunk.to_csv(output_file, index=False, mode='w')
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False)
        
        chunks_processed += 1
        total_rows += len(chunk)
        print(f"Processing... {total_rows:,} rows")
    
    print("-" * 50)
    print(f"Done! Total: {total_rows:,} rows saved to {output_file}")


if __name__ == "__main__":
    INPUT_FILE = "company_acquiror_patent_utility_info_202512291459.csv"
    OUTPUT_FILE = "output.csv"
    
    filter_columns(INPUT_FILE, OUTPUT_FILE)
