import os
import sys

# Ensure the 'src' package is discoverable
sys.path.append(os.path.abspath('.'))

from src.data.loader import load_data

def simulate():
    print("Simulating data generation process...")
    
    # Passing a non-existent file path will trigger the synthetic generation
    df = load_data('non_existent_dataset.csv')
    
    print("\n" + "="*50)
    print("--- First 5 rows ---")
    print(df.head())
    
    print("\n" + "="*50)
    print("--- Target Class Distribution ---")
    print(df['target'].value_counts())

if __name__ == '__main__':
    simulate()
