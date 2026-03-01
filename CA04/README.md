ML\_CA04 – Ensemble Models (Census Income Classification)



Overview



This project implements and evaluates multiple Ensemble Learning Models using the Census Income dataset previously used in CA03 for a Decision Tree Algorithm.



The objective is to:



* Recreate full data preparation pipeline
* Apply ensemble algorithms
* Tune key hyperparameter (`n\_estimators`)
* Compare model performance using Accuracy and AUC
* Identify optimal estimator values
* Provide final model comparison analysis





Program Structure



As outlined in the project structure diagram , the notebook follows a modular workflow:





│──  MAIN PROGRAM

├── FUNCTION 1: Data Preparation \& Reproducibility

├── FUNCTION 2: Random Forest Modeling + Analysis

├── FUNCTION 3: AdaBoost Modeling + Analysis

├── FUNCTION 4: Gradient Boost Modeling + Analysis

├── FUNCTION 5: XGBoost Modeling + Analysis

└── FUNCTION 6: Performance Comparison + Final Testing





Data Source



Dataset loaded directly from GitHub: https://github.com/ArinB/MSBA-CA-03-Decision-Trees/blob/master/census\_data.csv?raw=true



Initial dataset shape:

(48,842 rows, 11 columns)



FUNCTION 1: Data Preparation \& Reproducibility

Remove Duplicate Records

Auto Data Quality Analysis detected:

40,012 duplicate rows (~80% of dataset)



After removal:

Final dataset shape: (8,830 rows, 11 columns). This prevents:

* Inflated model performance
* Overfitting
* Distorted class distribution



Train/Test Split (Using Provided `flag` Column)

Dataset split according to predefined column:

Train: 5,106 rows

Test: 3,724 rows



The `flag` column was dropped after splitting to prevent data leakage.



Ordinal Encoding (Ordered Binned Variables). The following features represent ordered bins:

1. `hours\_per\_week\_bin`
2. `age\_bin`
3. `education\_num\_bin`
4. `capital\_gl\_bin`
5. `msr\_bin`



They were ordinal encoded (a=1, b=2, c=3, d=4, e=5).



One-Hot Encoding (Nominal Variables). The following nominal features were one-hot encoded:

1. &nbsp;`occupation\_bin`
2. &nbsp;`race\_sex\_bin`
3. `workclass\_bin`
4. &nbsp;`education\_bin`



Final feature matrix:

Train: (5106, 16)

Test:  (3724, 16)



All predictors are fully numeric and model-ready.



FUNCTION 2: Random Forest Modeling

Hyperparameter Tuning – `n\_estimators`



Tested values: \[50,100,150,200,250,300,350,400,450,500]



Observations:

Accuracy and AUC steadily increase up to ~350 estimators

Performance plateaus after 350–400 estimators

Optimal Range: 350–450 estimators





Final Random Forest Model

n\_estimators = 450



Final Performance:

Accuracy: 0.7307

AUC: 0.7803



FUNCTION 3: AdaBoost Modeling

Hyperparameter Tuning – `n\_estimators`



Observations:

Rapid improvement between 50–250 estimators

Accuracy peaks at ~250

AUC stabilizes around 200–300

Slight decline beyond 300 (mild overfitting behavior)



Optimal Range: 250–300 estimators



Final AdaBoost Model

n\_estimators = 250



Final Performance:

Accuracy: 0.7581

AUC: 0.8080



AdaBoost outperformed Random Forest on both metrics.



FUNCTION 4: Gradient Boosting



Observations:

Best performance between 100–150 estimators

Decline beyond 200 estimators

Shows earlier overfitting behavior



Optimal Range: 100–150 estimators



Peak AUC ≈ 0.811–0.812



FUNCTION 5: XGBoost

Tested same estimator range.



Observations:

Strong early performance

Gradual decline as estimators increase

Slight overfitting trend at higher values



Final Performance Comparison



| Model                | Accuracy    | AUC         | Stability   |

| -------------------- | ----------- | ----------- | ----------- |

| Decision Tree (CA03) | 0.75        | 0.77        | Moderate    |

| Random Forest        | 0.7307      | 0.7803      | Very Stable |

| AdaBoost             | 0.7581      | 0.8080      | Moderate    |

| Gradient Boost       | 0.754       | 0.812       | Sensitive   |

| XGBoost              |  0.7430     | 0.8020      | Efficient   |





Key Insights

1\. Removing duplicates significantly improved dataset integrity.

2\. Boosting models (AdaBoost \& Gradient Boost) achieved higher AUC than Random Forest.

3\. Performance gains plateau beyond certain estimator thresholds.

4\. Increasing estimators indefinitely does not guarantee better performance.

5\. AdaBoost provided the best balance between Accuracy and AUC.



Conceptual Takeaways

Random Forest → Reduces variance (Bagging)

AdaBoost / Gradient Boost → Reduces bias (Boosting)

Optimal `n\_estimators` varies by algorithm

Early stopping region is often ideal



How to Run

1\. Clone repository

2\. Open notebook in Google Colab

3\. Run cells sequentially

4\. Review graphs and summary output


Libraries (Dependencies)

* numpy
* pandas
* scikit-learn
* matplotlib
* xgboost





