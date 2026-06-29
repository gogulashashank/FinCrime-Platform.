import os
import sys

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.pipeline import FeaturePipeline
from src.models.rules import RuleEngine
import polars as pl

def main():
    print("--- Starting Milestone 1.2 Execution ---")
    
    # 1. Feature Engineering
    pipeline = FeaturePipeline()
    pipeline.load_data()
    base_table = pipeline.engineer_features()
    pipeline.save_features()
    
    # 2. Rule Engine Scoring
    engine = RuleEngine()
    alerts = engine.evaluate(base_table)
    
    # Save Alerts
    output_dir = 'data/synthetic'
    alerts_path = os.path.join(output_dir, 'fact_alerts.parquet')
    alerts.write_parquet(alerts_path)
    print(f"Saved {len(alerts)} alerts to {alerts_path}")
    
    # Save a sample for GitHub
    sample_path = 'data/sample_seed/fact_alerts_sample.csv'
    alerts.head(100).write_csv(sample_path)
    print(f"Saved sample alerts to {sample_path}")

    print("--- Milestone 1.2 Execution Complete ---")

if __name__ == "__main__":
    main()
