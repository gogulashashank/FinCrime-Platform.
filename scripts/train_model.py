import os
import sys
import polars as pl

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ml_detector import MLDetector

def main():
    print("--- Starting ML Model Training ---")
    
    # Load the analytical base table (which contains our 'is_suspicious' labels)
    data_path = 'data/synthetic/analytical_base_table.parquet'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run the feature pipeline first.")
        return
        
    base_table_pl = pl.read_parquet(data_path)
    base_table_pd = base_table_pl.to_pandas()
    
    # Train and Explain
    detector = MLDetector()
    detector.train(base_table_pd)
    
    print("--- Running Inference & Explanations on full dataset ---")
    scored_df = detector.predict_and_explain(base_table_pd)
    
    # Save the ML scored results
    output_path = 'data/synthetic/ml_scored_transactions.parquet'
    scored_df.to_parquet(output_path, index=False)
    print(f"Saved ML scored dataset to {output_path}")

if __name__ == "__main__":
    main()
