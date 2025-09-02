#!/usr/bin/env python3
"""
Script to measure accuracy on rebus puzzle datasets using the accuracy_mesurement function.
This script tests the performance of different models on the provided datasets.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src import utils

def main():
    """Main function to measure accuracy on datasets."""
    
    print("Rebus Puzzle Accuracy Measurement")
    print("=" * 50)
    
    # Check if API keys are set
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('GOOGLE_API_KEY'):
        print("Error: No API keys found!")
        print("Please set OPENAI_API_KEY or GOOGLE_API_KEY environment variables.")
        print("You can create a .env file with:")
        print("OPENAI_API_KEY=your_key_here")
        print("GOOGLE_API_KEY=your_key_here")
        return
    
    # Define datasets to test
    datasets = {
        "mixed_difficulty_set": "../datasets/mixed_difficulty_set.json",
        "o3_mini_labeled": "../datasets/o3_mini_labeled.json"
    }
    
    # Define models to test
    models = {
        #"o3-mini": "OpenAI",
        #"o3": "OpenAI", 
        "gpt-4o": "OpenAI",
        "gemini-2.0-flash": "Google"
    }
    
    results = {}
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n Testing Dataset: {dataset_name}")
        print("-" * 30)
        
        try:
            # Load dataset
            dataset = utils.load_json_by_mode(dataset_path, mode = "dataframe")[:10]
            print(f" Loaded dataset with {len(dataset['Problems'])} problems")
            
            dataset_results = {}
            print(dataset)
            for model_name, provider in models.items():
                print(f"\n Testing Model: {model_name} ({provider})")
                print("This may take several minutes...")
                
                try:
                    # Measure accuracy
                    accuracy = utils.accuracy_mesurement(
                        dataset=dataset,
                        model=model_name,
                        provider=provider
                    )
                    
                    dataset_results[model_name] = accuracy
                    print(f" Accuracy: {accuracy:.2f}%")
                    
                except Exception as e:
                    print(f" Error with {model_name}: {e}")
                    dataset_results[model_name] = None
            
            results[dataset_name] = dataset_results
            
        except Exception as e:
            print(f" Error loading dataset {dataset_name}: {e}")
            results[dataset_name] = {}
    
    # Print summary
    print("\n" + "=" * 50)
    print(" ACCURACY SUMMARY")
    print("=" * 50)
    
    for dataset_name, dataset_results in results.items():
        print(f"\n {dataset_name.upper()}:")
        print("-" * 20)
        
        if not dataset_results:
            print(" No results available")
            continue
            
        for model_name, accuracy in dataset_results.items():
            if accuracy is not None:
                print(f"  {model_name}: {accuracy:.2f}%")
            else:
                print(f"  {model_name}: Failed")
    
    # Save results to file
    try:
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"accuracy_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for dataset_name, dataset_results in results.items():
            serializable_results[dataset_name] = {}
            for model_name, accuracy in dataset_results.items():
                if accuracy is not None:
                    serializable_results[dataset_name][model_name] = float(accuracy)
                else:
                    serializable_results[dataset_name][model_name] = None
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n Results saved to: {results_file}")
        
    except Exception as e:
        print(f" Error saving results: {e}")

if __name__ == "__main__":
    main() 