# Code Infilling Project for Conditional Statements

This project focuses on fine-tuning a BERT model to perform code infilling, specifically targeting conditional `if` statements in Python functions. The model learns to predict missing `if` conditions, enhancing its capability to understand and complete code syntax accurately.

## Table of Contents
1. [Project Objective](#project-objective)
2. [Dataset Construction](#dataset-construction)
3. [Model Training](#model-training)
4. [Inference](#inference)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [File Structure](#file-structure)
8. [Setup Instructions](#setup-instructions)
9. [Usage](#usage)
10. [Contact Information](#contact-information)

---

## Project Objective
The primary goal of this project is to train a BERT-based model that can infill masked `if` statements within Python code, predicting logical conditions in partially completed functions. By fine-tuning the model on a dataset of code snippets with masked `if` statements, the model learns syntax patterns and context, enabling it to generate missing code accurately.

## Dataset Construction
The dataset used in this project was derived from [CodeSearchNet](https://github.com/github/CodeSearchNet) and other open-source repositories. The key steps in dataset preparation include:
1. **Filtering**: Selecting code snippets with conditional `if` statements.
2. **Masking**: Masking the `if` statements to create infill tasks for the model.
3. **Tokenization**: Tokenizing the snippets using a pre-trained BERT tokenizer, with padding and truncation to a maximum length of 512 tokens.

The processed dataset is split into training and evaluation sets.

## Model Training
A pre-trained BERT model was fine-tuned using the masked dataset, where the model learns to predict the missing `if` condition. The training configuration includes:
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3

### Training Script
The training script is located in the `scripts` directory:
```bash
python scripts/run_finetuning.py
```

This script loads the dataset, fine-tunes the model, and saves checkpoints in the `output_model_hf` directory.

## Inference
After training, the model can be used to infill masked `if` conditions in new code snippets.

### Inference Script
Run the following command to perform inference on a sample:
```bash
python scripts/inference.py
```
The script loads the fine-tuned model and infills the masked `if` statements in given code snippets. The predictions are stored with confidence scores.

## Evaluation
The model's performance is evaluated on two test sets:
1. **Custom Test Set** (`generated-testset.csv`): Created by masking `if` statements in various functions.
2. **Provided Test Set** (`provided-testset.csv`): A predefined set for standardized evaluation.

Each test set includes:
- **Input Code**: The code snippet with a masked `if` condition.
- **Expected Condition**: The correct `if` condition.
- **Predicted Condition**: The model's predicted condition.
- **Prediction Score**: A score from 0-100 representing the model's confidence.

## Results
The model's predictions and confidence scores are saved in CSV files for both test sets. These files include columns for input, expected condition, predicted condition, and prediction score.

### Sample Result
```plaintext
Input,Correct (True/False),Expected if condition,Predicted if condition,Prediction score (0-100)
"def check_value(x): [MASK] x > 10: return 'High' else: return 'Low'",False,if,[MASK],10.25
```

## File Structure
```plaintext
HW2_CSCI_680/
├── data/                   # Raw and processed data files
├── lightning_logs/         # Training logs
├── models/                 # Pre-trained and fine-tuned models
├── output_model/           # Fine-tuned model checkpoints
├── output_model_hf/        # Hugging Face model output
├── results/                # Results and evaluation output files
├── scripts/                # All Python scripts for processing, training, and inference
│   ├── run_finetuning.py       # Script for model training
│   ├── inference.py            # Script for model inference
│   ├── prepare_finetuning_data.py  # Script for dataset preparation
│  
└── tokenizer/              # Tokenizer files
└── provided-testset.csv  
└── generated-testset.csv   # CSV for generated test set results
```

## Setup Instructions


1. **Install Dependencies**
   Create a virtual environment and install required packages:
   ```bash
   conda create -n csci_680 python=3.10
   conda activate csci_680
   pip install -r requirements.txt
   ```

2. **Download Pre-trained BERT Model and Tokenizer**
   Make sure the BERT model and tokenizer files are in the `tokenizer/` and `models/` directories.

## Usage

### Training
To train the model, use:
```bash
python scripts/run_finetuning.py
```

### Inference
To test the model's predictions on new code snippets, use:
```bash
python scripts/inference.py
```

### Evaluation
Evaluation results for the model's performance on both test sets are saved in `generated-testset.csv` and `provided-testset.csv`.

## Contact Information
For questions or feedback, please reach out to:
- **Md Zahidul Haque Alvi**  
- Email: [mhaque@wm.edu](mailto:mahque@wm.edu)  

