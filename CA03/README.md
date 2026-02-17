CA03_ML – Decision Tree Classification Project

Content Guide

PSD structure	---------------------------	page 1 
Data cleaning process ---------------------	pages 3–8 
Modeling workflow -------------------------	pages 9–16
Final evaluation & prediction sections ---- 	pages 17–18


Project Overview
This project implements a Decision Tree Classifier (CART algorithm) to predict income classification (`<=50K` vs `>50K`) using a structured, categorical census dataset.

The implementation follows a structured functional design:

1. MAIN PROGRAM
   ↓
2. Build Baseline Model
   ↓
3. Hyperparameter Tuning
   ↓
4. Visualize Best Tree
   ↓
5. Predict New Individual
   ↓
END PROGRAM


The workflow strictly follows the program structure diagram provided in the assignment .

Data Source
Dataset loaded directly from GitHub:https://github.com/ArinB/MSBA-CA-03-Decision-Trees/blob/master/census_data.csv?raw=true

Dataset Characteristics
- 48,842 rows
- 11 columns
- Binary classification problem (`y`)
- All predictors are categorical (pre-binned)
- No continuous numeric features


Step 1 – Data Understanding & EDA

AutoViz was used for automated EDA.

Key Findings:
- 40,012 duplicate rows (~82%)
- No missing values
- `flag` column indicates train/test split (not predictive)
- Several ordinal categorical features were incorrectly treated as nominal


Step 2 – Data Cleaning & Transformation

Remove Duplicate Records
- Reduced dataset from 48,842 → 8,830 rows
- Prevented inflated accuracy and overfitting risk

Train/Test Split (Using Provided Flag)

Dataset split based on `flag` column:
- Train: 5,106 rows
- Test: 3,724 rows

`flag` column dropped afterward to prevent data leakage.

Ordinal Encoding (Ordered Bins). Applied to:
- `hours_per_week_bin`
- `age_bin`
- `education_num_bin`
- `capital_gl_bin`
- `msr_bin`

Converted:
a. → 1
b. → 2
c. → 3
d. → 4
e. → 5
This preserves ranking information.


One-Hot Encoding (Nominal Variables)
Applied to:
- `occupation_bin`
- `race_sex_bin`
- `workclass_bin`
- `education_bin`

Used python, pd.get_dummies(..., drop_first=True)

Final dataset:
- 16 numeric features
- Fully model-ready

Feature Selection (Correlation Check)
Correlation matrix evaluated on training set.

Result:
- No multicollinearity issues detected
- No features removed


Step 3 – Decision Tree Modeling

Baseline Model

python
DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=15,
    max_features=None,
    random_state=101
)


Baseline Performance
--------------------------------
| Metric              | Value  |
--------------------------------
| Accuracy            | 0.7489 |
| Precision (Class 1) | 0.64   |
| Recall (Class 1)    | 0.44   |
| F1 Score (Class 1)  | 0.52   |
--------------------------------

Class Imbalance Observation

Class Distribution (after cleaning):

- Class 0 (<=50K): Majority
- Class 1 (>50K): Minority

Model shows bias toward majority class.


Step 4 – Hyperparameter Tuning

Four tuning experiments conducted:
1. Split Criterion (`gini` vs `entropy`)
2. `min_samples_leaf`
3. `max_features`
4. `max_depth`

Best Hyperparameters Found

python
criterion = "entropy"
min_samples_leaf = 15
max_features = 0.5
max_depth = 16
random_state = 101


Best Model Performance
-------------------------
| Metric    | Value     |
-------------------------
| Accuracy  | 0.7535    |
| Recall    | 0.4042    |
| Precision | 0.6686    |
| F1 Score  | 0.5038    |
| Runtime   | 0.011 sec |
-------------------------

Interpretation
- Slight accuracy improvement
- Recall for minority class still modest (~40%)
- Model does NOT show overfitting (Train = Test accuracy)


Decision Tree Visualization
Tree was exported using:python, export_graphviz() And rendered via Graphviz.

Visualization confirms:
- Most important splits appear at top levels
- Depth controlled via `max_depth`
- Overgrowth prevented via `min_samples_leaf`

Overfitting Assessment
Train Accuracy: ~74%
Test Accuracy: ~75%

Conclusion:
- Model is not overfitting
- Generalization is stable
- Controlled complexity prevents full growth


Step 5 – Predict New Individual
New data row:
- Encoded using same ordinal + one-hot logic
- Reindexed to match training feature columns
- Passed into trained model

Example output:
python
Prediction: >50K
Probability: 0.7692

Key Observations
1. Duplicate removal significantly changed class distribution.
2. Proper ordinal encoding preserved meaningful ranking.
3. One-hot encoding prevented artificial ordering.
4. Hyperparameter tuning improved model performance.
5. Minority class recall remains the main limitation.
6. Tree is controlled and interpretable.


Project Structure
├── Data Loading
├── Auto EDA
├── Data Quality Report
├── Duplicate Removal
├── Train/Test Split
├── Ordinal Encoding
├── One-Hot Encoding
├── Feature Selection
├── Baseline Decision Tree
├── Hyperparameter Tuning
├── Model Evaluation
├── Tree Visualization
├── New Prediction Example

How to Run
1. Open notebook in Google Colab
2. Run cells sequentially
3. Install Graphviz when prompted
4. Review evaluation outputs
5. Inspect visualized tree

Final Conclusion
This project demonstrates:
- Proper structured ML pipeline design
- Data quality auditing before modeling
- Encoding strategy aligned with feature types
- Controlled decision tree modeling
- Hyperparameter optimization via systematic experimentation
- Interpretability via tree visualization

The final model achieves about 75% accuracy with controlled complexity and stable generalization.

