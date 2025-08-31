# Quick Start Guide

This guide will help you get up and running with the Rebus Generation Package in under 5 minutes.

## Prerequisites

- Python 3.9 or higher
- API key for OpenAI or Google AI services

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API keys:**
   Create a `.env` file in the package root:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   # OR
   echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
   ```

## Step 2: Basic Usage

###  Offline Puzzle Generation
Run the basic example script in examples:
```bash
python example_usage_reasoning.py
```

This will:
- Load the word bank and domains
- Generate rebus puzzles
- Display the puzzles and solutions

## Step 3: Custom Usage

### Generate Puzzles Programmatically

```python
from src import offline_generation, utils

# Load data
domains = utils.load_json_by_mode("datasets/domains.json")
with open("words.txt", "r") as f:
    words = f.read().replace('\n', ' ').split(" ")

# Generate puzzles
# the threshold can be an integer maing the character length of easy and medium the same.
# It can be a tupple so that the first number is the highest length for low, and the second number is the highest length for medium
puzzles = offline_generation.generate_bank_reasoning(
    domains=domains,
    words=words,
    system_prompt="Your custom prompt here",
    thresholds=6,
    num_samples=5,
    model="o3-mini"
)

# Print results
for puzzle in puzzles:
    print(f"Question: {puzzle['generated_question']}")
    print(f"Solution: {puzzle['generated_solution']}")
```

### Generate ICL Examples

```python
from src import icl_generation

icl_generation.generate_icl_examples(
    domains=domains,
    words=words,
    minimum_problem_per_difficulty=10
)
```
## Troubleshooting

### Common Issues

1. **"No API keys found"**
   - Make sure your `.env` file exists and contains valid API keys
   - Check that the keys are properly formatted

2. **"File not found" errors**
   - Ensure you're running scripts from the package root directory
   - Check that all data files are present

3. **Import errors**
   - Make sure you've installed all dependencies: `pip install -r requirements.txt`
   - Verify you're using Python 3.9+

4. **API rate limits**
   - The package includes retry logic, but you may need to wait between requests
   - Consider using a model with higher rate limits

### Getting Help

1. Check the main `README.md` for detailed documentation
2. Review the source code in the `src/` directory
3. Examine the example notebooks in `examples/`

## Next Steps

- Experiment with different models and difficulty levels
- Customize the system prompts for your specific use case
- Generate larger datasets for research purposes
- Explore the ICL examples to understand the format

Happy puzzle generating! 🎯 
