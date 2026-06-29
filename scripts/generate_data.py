import os
import sys

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.generator.data_generator import FinCrimeDataGenerator

def main():
    print("Initializing FinCrime Synthetic Data Generator...")
    generator = FinCrimeDataGenerator(num_entities=2000, num_transactions=100000, seed=42)
    
    print("Generating Entities...")
    generator.generate_entities()
    
    print("Generating Accounts...")
    generator.generate_accounts()
    
    print("Generating Transactions (with injected typologies)...")
    generator.generate_transactions()
    
    print("Saving output datasets...")
    generator.save_data()
    
    print("Done! Check data/synthetic and data/sample_seed.")

if __name__ == "__main__":
    main()
