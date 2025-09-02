#!/usr/bin/env python3
"""
Simple example script demonstrating basic usage of the rebus generation package.
This script shows how to generate rebus puzzles without ICL examples.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src import offline_generation, utils, prompts

def main():
    """Main function demonstrating basic rebus generation."""
    
    print(" Rebus Generation Package - Basic Example")
    print("=" * 50)
    
    # Check if API keys are set
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('GOOGLE_API_KEY'):
        print(" Error: No API keys found!")
        print("Please set OPENAI_API_KEY or GOOGLE_API_KEY environment variables.")
        print("You can create a .env file with:")
        print("OPENAI_API_KEY=your_key_here")
        print("GOOGLE_API_KEY=your_key_here")
        return
    
    try:
        # Load domains
        print(" Loading domains...")
        domains = utils.load_json_by_mode("../datasets/domains.json")
        print(f" Loaded {len(domains)} domains")
        
        # Load words
        print(" Loading word bank...")
        with open("words.txt", "r") as f:
            words_data = f.read()
        words = words_data.replace('\n', ' ').split(" ")
        words = [word for word in words if word.strip()]  # Remove empty strings
        words = words[9770:]
        print(f" Loaded {len(words)} words")
        
        # Generate a simple system prompt
        print(" Creating system prompt...")
        system_prompt = prompts.construct_system_prompt(domains=domains, words=words, example_directory="../ICL-examples")
        # Generate puzzles
        print(" Generating rebus puzzles...")
        print("This may take a few minutes depending on your model choice...")
        
        puzzles = offline_generation.generate_bank_reasoning(
            domains=domains,
            words=words,
            system_prompt=system_prompt,
            thresholds=6,  # Easy difficulty (≤6 characters)
            num_samples=3,  # Generate 3 puzzles
            model="o3-mini",  # Use Claude 3.5 Haiku
            model_provider="OpenAI"
        )
        
        # Display results
        print("\n Generated Puzzles:")
        puzzles.to_json("offline_generation.json")
        print("=" * 50)
        for i in range(len(puzzles)):
            print(f"\nPuzzle {i}:")
            print(f"Question: {puzzles['Problems'][i]}")
            print(f"Solution: {puzzles['Solutions'][i]}")
            print(f"Difficulty: {puzzles['Difficulty Level'][i]}")
            print("-" * 30)
        
        print(f"\n Successfully generated {len(puzzles)} puzzles!")
        
    except FileNotFoundError as e:
        print(f" File not found: {e}")
        print("Make sure you're running this script from the package root directory.")
    except Exception as e:
        print(f" Error occurred: {e}")
        print("Check your API keys and internet connection.")

if __name__ == "__main__":
    main() 