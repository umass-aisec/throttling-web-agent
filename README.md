# Throttling Web Agents Using Reasoning Gates

## Demo - https://aisec.cs.umass.edu/demo/web-agent-throttling/




A Python package for generating reasoning-based rebus puzzles using large language models. This package supports both offline generation of rebus puzzles and in-context learning (ICL) example generation for improved model performance.

## Overview

This package provides tools to:
- Generate rebus puzzles with varying difficulty levels (easy, medium, hard)
- Create in-context learning (ICL) examples for improved model performance
- Support multiple AI models (OpenAI GPT models, Google Gemini models)
- Control puzzle difficulty based on solution word length
- Generate puzzles across diverse academic domains

## Installation

### Prerequisites
- Python 3.9 or higher
- API keys for OpenAI and/or Google AI services

### Setup

1. **Install dependencies:**
   ```bash
   pip install openai>=1.0.0 python-dotenv>=1.0.0 pandas requests beautifulsoup4 tqdm tenacity google-genai
   ```

2. **Set up environment variables:**
   Create a `.env` file in the package root with your API keys:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```
3. **Set up toml:**
   ```bash
   pip install -U pip setuptools wheel 
   pip install -e .
   ```

   

## Package Structure

```
cleaned_version_complete_1/
├── src/                    # Core package modules
│   ├── __init__.py
│   ├── icl_generation.py   # ICL example generation
│   ├── offline_generation.py # Offline puzzle generation
│   ├── prompts.py          # System prompts and utilities
│   └── utils.py            # Utility functions and API calls
├── examples/               # Usage examples
│   ├── example_usage_non_reasoning.py
│   ├── example_usage_reasoning.py
│   └── words.txt          # Word bank for puzzle generation
├── ICL-examples/          # Generated ICL examples
│   ├── high_reasoning.json
│   ├── medium_reasoning.json
│   ├── low_reasoning.json
│   └── ...
├── datasets/              # Domain and test datasets
│   ├── domains.json       # Academic domains for puzzle themes
│   ├── mixed_difficulty_set.json
│   └── ...
└── pyproject.toml         # Package configuration
```

## Core Functionality

### 1. Offline Generation

Generate rebus puzzles without using ICL examples. This is useful for models that don't benefit from in-context learning.

**Key Functions:**
- `generate_bank_reasoning()`: Generate reasoning-based puzzles
- `generate_bank_non_reasoning()`: Generate non-reasoning puzzles

**Example Usage:**
```python
from src import offline_generation
from src import utils

# Load domains and words
domains = utils.load_json_by_mode("datasets/domains.json")
with open("words.txt", "r") as f:
    words = f.read().replace('\n', ' ').split(" ")

# Generate reasoning-based puzzles
puzzles = offline_generation.generate_bank_reasoning(
    domains=domains,
    words=words,
    system_prompt="Your system prompt here",
    thresholds=6,  # Difficulty threshold
    num_samples=10,
    model="o3-mini",
    model_provider="OpenAI"
)
```

### 2. ICL Example Generation

Generate in-context learning examples to improve model performance on rebus generation tasks.

**Key Function:**
- `generate_icl_examples()`: Generate ICL examples with difficulty control

**Example Usage:**
```python
from src import icl_generation
from src import utils

# Load domains and words
domains = utils.load_json_by_mode("datasets/domains.json")
with open("words.txt", "r") as f:
    words = f.read().replace('\n', ' ').split(" ")

# Generate ICL examples
icl_generation.generate_icl_examples(
    domains=domains,
    words=words,
    minimum_problem_per_difficulty=10
)
```

### 3. System Prompt Construction

Build system prompts that incorporate ICL examples for improved performance.

**Example Usage:**
```python
from src import prompts

# Construct system prompt with ICL examples
system_prompt = prompts.construct_system_prompt(
    domains=domains,
    example_directory="ICL-examples"
)
```

## Supported Models

### OpenAI Models tested 
- `gpt-4o`
- `o3`
- `o3-mini`

### Google Models tested
- `gemini-2.5-flash`
- `gemini-2.5-pro`
- `gemma-3-27b-it`

## Difficulty Control

During offline generation you can provide difficulty control length as amn integer or a tupple:

- if integer us provided
  
   - **Easy**: Solutions ≤ integer
   - **Medium**: Solutions ≤ integer
   - **Hard**: Solutions ≥ integer
     
- if tupple us provided
  
   - **Easy**: Solutions ≤ first value of tupple
   - **Medium**: Solutions ≥ first value of tupple and Solutions ≤  second value of tupple
   - **Hard**: Solutions ≥ second value of tupple  

You can customize thresholds by passing different values to the generation functions.

## Data Sources

### Domains
The package several domains domains covering:
- Sciences (Physics, Chemistry, Biology, etc.)
- Engineering disciplines
- Humanities (History, Philosophy, Literature, etc.)
- Social sciences (Psychology, Sociology, Economics, etc.)
- Arts and creative fields
- And many more...

### Word Bank
- Contains over 370,000 English words
- Sorted by length for difficulty control
- Filtered for appropriate puzzle generation

## Usage Examples

### Basic Puzzle Generation for non-reasoning models
```python
# Simple puzzle generation without ICL
from src import offline_generation
from src import utils

domains = utils.load_json_by_mode("datasets/domains.json")
with open("words.txt", "r") as f:
    words = f.read().replace('\n', ' ').split(" ")

# Generate 5 easy puzzles
easy_puzzles = offline_generation.generate_bank_non_reasoning(
    domains=domains,
    words=words,
    num_samples=5
)
```

### Advanced ICL-Based Generation
```python
# Generate ICL examples first
from src import icl_generation
icl_generation.generate_icl_examples(
    domains=domains,
    words=words,
    minimum_problem_per_difficulty=5
)

# Then use ICL-enhanced system prompt
from src import prompts
system_prompt = prompts.construct_system_prompt(
    domains=domains,
    example_directory="ICL-examples"
)

# Generate puzzles with ICL examples
puzzles = offline_generation.generate_bank_reasoning(
    domains=domains,
    words=words,
    system_prompt=system_prompt,
    thresholds=6,
    num_samples=10,
    model="gpt-4o"
)
```

### Custom Difficulty Control
```python
# Custom thresholds for difficulty control
custom_thresholds = (4, 8)  # Medium difficulty: 4-8 characters

puzzles = offline_generation.generate_bank_reasoning(
    domains=domains,
    words=words,
    system_prompt=system_prompt,
    thresholds=custom_thresholds,
    num_samples=20,
    model="o3-mini"
)
```

## Output Format

Generated puzzles include:
- `generated_question`: The rebus puzzle text
- `generated_solution`: The correct answer
- `labels`: Difficulty classification

## Error Handling

The package includes robust error handling for:
- API rate limits and timeouts
- Invalid puzzle generation
- Model-specific issues
- File I/O operations

## License

This package is part of a research project.
## Support

For issues and questions:
1. Check the examples in the `examples/` directory
2. Review the source code in `src/`
3. Ensure your API keys are properly configured
4. Verify your Python environment meets the requirements

## Citation
```
@misc{kumar2025throttlingwebagentsusing,
  title={Throttling Web Agents Using Reasoning Gates}, 
  author={Abhinav Kumar and Jaechul Roh and Ali Naseh and Amir Houmansadr and Eugene Bagdasarian},
  year={2025},
  eprint={2509.01619},
  url={https://arxiv.org/abs/2509.01619}
}

```
