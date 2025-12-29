# Nationality Bias Detection in Large Language Models

**Bachelor's Thesis Research Project**

## Abstract

Large language models are increasingly used in settings where people are referenced by name, including hiring, customer support, education, and health. This research investigates nationality-linked variation in model behavior by measuring how token log-probabilities differ when swapping names from different countries in otherwise identical contexts.

We study two open-source 7B chat models: **LLaMA-2-7B-chat** and **Mistral-7B-Instruct-v0.3**. Using 50 natural prompts across Job, Education, Health, and Investment domains, we systematically replace a placeholder name with country-specific full names from five countries (United Kingdom, India, Turkey, Germany, China), using 20 names per gender and country (10,000 prompt-name pairs per model).

Both models show systematic nationality differences. In LLaMA-2, German and Turkish names have substantially higher perplexity than other countries. In Mistral, Turkish names show highest perplexity, followed by Indian names, with British names lowest. Gender and prompt category further modulate these gaps, with job prompts generally producing higher perplexities.

## Methodology

### Dataset
- **Base Prompts**: 50 natural language prompts derived from the RealWorldQuestioning dataset
- **Domains**: Job, Education, Health, Investment
- **Countries**: United Kingdom, India, Turkey, Germany, China
- **Names per Country**: 20 male and 20 female names from `name-dataset`
- **Total Combinations**: 10,000 prompt-name pairs per model

### Measurement
- Token log-probabilities scored via `llama.cpp`
- Name-span perplexity computed from mean log-likelihood
- Names matched to fixed token length to reduce tokenization confounds
- Surname repetition constrained for consistency

### Models
- **LLaMA-2-7B-chat** (Meta)
- **Mistral-7B-Instruct-v0.3** (Mistral AI)

## Repository Structure

```
nationality_bias_detection.py    # Main experimental script for bias detection
questions.csv                     # Dataset of prompts used in experiments
```

## Key Findings

- **Model-Dependent Disparities**: Direction and magnitude of nationality bias vary strongly between models
- **Systematic Differences**: Both models show consistent nationality-based perplexity variations
- **Domain Effects**: Job-related prompts generally produce higher perplexities
- **Gender Interactions**: Gender further modulates nationality-based gaps

## Usage

The main experimental script (`nationality_bias_detection.py`) implements the bias detection methodology, including:
- Name substitution in prompts
- Token-level perplexity calculation
- Statistical analysis across nationalities, genders, and domains

```bash
python nationality_bias_detection.py
```

## Requirements

- Python 3.x
- `llama.cpp` for model inference
- Standard data science libraries (numpy, pandas, etc.)

## Citation

If you use this work, please cite:

```
[Citation details to be added upon thesis publication]
```

## Acknowledgments

This research was conducted as part of a Bachelor's thesis. Prompts adapted from the RealWorldQuestioning dataset (Prabhune et al., 2025). Names sourced from `name-dataset`.

## License

[License information to be added]
