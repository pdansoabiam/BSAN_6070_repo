CA02 Improved Naive Bayes Email Spam Classifier

This repository contains an improved design and implementation of an email spam classifier using the Naive Bayes supervised learning algorithm.
The implementation addresses efficiency, scalability, robustness, and reproducibility weaknesses present in the original starter code, while preserving the original assignment intent and framework.

Improved Design Overview

The improved solution follows this end-to-end pipeline:
Data → Vocabulary → Feature Matrix + Labels → Train Naive Bayes → Test & Evaluate → Human-Readable Output

Key goals of the redesign:
Improve computational efficiency
Improve data handling and robustness
Improve clarity, interpretability, and reproducibility
Align with standard machine-learning practices

Key Improvements Over Original Code
1. Efficient Feature Construction
Uses `Counter` for single-pass word counting (O(n))
Uses a word-to-index dictionary for O(1) lookup
Eliminates nested loops over the vocabulary
Replaces dense NumPy arrays with sparse matrices (CSR)

2. Improved Text Processing
Reads the entire email, not just a single line
Converts text to lowercase for normalization
Uses regex-based tokenization for consistency
Drops very short tokens to reduce noise
Handles encoding safely to avoid crashes

3. Robust File and Path Handling
Uses `pathlib.Path` for platform-independent paths
Sorts file lists for reproducible ordering
Uses filename-based labeling in a dedicated function

4. Cleaner Modeling Workflow
Vocabulary built only from training data
Same vocabulary applied to both train and test sets
Uses `MultinomialNB`, appropriate for word-count features
Evaluates performance with accuracy, confusion matrix, and classification report

Feature Representation
Each email is represented as a 3000-dimensional bag-of-words vector
Each vector position corresponds to a vocabulary word
The value is the count of that word in the email
Feature matrices are stored as sparse matrices for efficiency

Model Performance (Test Set)
Accuracy: 96.15%
Confusion Matrix:

---
[[129   1]
 [  9 121]]
---

Precision, recall, and F1-scores are balanced across classes, indicating strong performance on both spam and non-spam emails.

Human-Readable Prediction Output
In addition to numeric predictions, the model produces a clear output table with:
Email filename
Actual label (Yes / No)
Predicted label (Yes / No)
Target_status column:

Yes → Spam
No → Not Spam

This improves interpretability and aligns predictions with real-world usage.

Repository Structure

---
CA02/
├── train-mails/          Training email files
├── test-mails/           Testing email files
├── pda_ah_CA02_NB_assignment.ipynb
├── README.md
└── spam_predictions.csv  Optional output file
---

Note; The notebook and the `train-mails` / `test-mails` folders must remain in the same directory so that relative paths (`./train-mails`, `./test-mails`) work correctly across different machines.

Technologies Used
Python
NumPy
Pandas
SciPy (Sparse Matrices)
scikit-learn (Multinomial Naive Bayes)

Final comment, this improved implementation;
Preserves the original CA02 assignment logic
Fixes known inefficiencies and design weaknesses
Produces accurate, reproducible, and interpretable results
Reflects industry-standard practices for text classification
