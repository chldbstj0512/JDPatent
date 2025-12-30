import pandas as pd


def extract_applicants_by_acquiror(input_file: str, output_file: str) -> None:
    """
    Group applicants by acquiror_short_name and save to a new CSV file.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Total rows: {len(df):,}")
    print(f"Unique acquiror_short_name: {df['acquiror_short_name'].nunique():,}")
    
    # Group by target_short_name and aggregate applicants
    result = df.groupby('acquiror_short_name').agg({
        'acquiror_nation': 'first',
        'applicant': lambda x: ', '.join(x.dropna().unique())
    }).reset_index()
    
    result.columns = ['acquiror_short_name', 'acquiror_nation', 'applicants']
    
    # Add count column
    applicant_count = df.groupby('acquiror_short_name')['applicant'].nunique().reset_index()
    applicant_count.columns = ['acquiror_short_name', 'applicant_count']
    
    result = result.merge(applicant_count, on='acquiror_short_name')
    
    # Sort by applicant count descending
    result = result.sort_values('applicant_count', ascending=False)
    
    # Save to CSV
    result.to_csv(output_file, index=False)
    
    print(f"Done! Saved {len(result):,} rows to {output_file}")


if __name__ == "__main__":
    INPUT_FILE = "output.csv"
    OUTPUT_FILE = "result.csv"
    
    extract_applicants_by_acquiror(INPUT_FILE, OUTPUT_FILE)

