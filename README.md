# Projects

# Credit Card Fraud Detection

This repository contains a machine learning project focused on detecting fraudulent credit card transactions. The project addresses the challenges of highly imbalanced datasets, a common characteristic in fraud detection scenarios.

## Project Overview

Credit card fraud detection is a critical application of machine learning. The goal is to build a robust model that can accurately identify fraudulent transactions while minimizing false positives, which can inconvenience legitimate cardholders. This project utilizes a dataset of credit card transactions, many of which are anonymized due to privacy concerns.

## Table of Contents

1.  [Understand the Problem & Data](#1-understand-the-problem--data)
2.  [Data Preprocessing](#2-data-preprocessing)
3.  [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4.  [Handle Class Imbalance](#4-handle-class-imbalance)
5.  [Model Selection](#5-model-selection)
6.  [Model Evaluation](#6-model-evaluation)
7.  [Results](#7-results)
8.  [How to Run the Code](#8-how-to-run-the-code)
9.  [Future Enhancements](#9-future-enhancements)

---

## 1. Understand the Problem & Data

The core of this project is a **binary classification problem**: distinguishing between fraudulent transactions (labeled '1') and legitimate transactions (labeled '0').

### Key Data Characteristics:

* **Class Imbalance**: The dataset exhibits significant class imbalance. As shown in the "Class Distribution" plot, there are approximately 280,000 non-fraudulent transactions and only around 500 fraudulent transactions. This means fraud constitutes a minuscule percentage (approximately 0.172%) of the total data. This imbalance is a major challenge, as a naive model might simply predict "non-fraudulent" for all transactions and still achieve high accuracy, while failing to detect any fraud.
* **Features**:
    * **PCA-transformed features (V1 to V28)**: These are abstract numerical features derived from applying Principal Component Analysis (PCA) on the original, confidential transaction data. PCA helps in dimensionality reduction and noise reduction while preserving the most important patterns for fraud detection and ensuring user privacy.
    * **Original features**:
        * `Time`: The time elapsed between the first transaction in the dataset and the current transaction.
        * `Amount`: The transaction amount.
    * **Target variable**: `Class` (0 for non-fraud, 1 for fraud).

## 2. Data Preprocessing

Data preprocessing is a crucial step to prepare the raw data for machine learning models.

* **Handling Missing or Inconsistent Data**: While not explicitly shown in the provided code snippets, a common initial step involves checking for and addressing any missing or inconsistent data points. For this dataset, it appears to be relatively clean.
* **Normalization or Scaling Features**:
    * The `Amount` feature, and potentially `Time` (though `Time` was dropped in this specific code due to shuffling and PCA), often have different scales compared to other features. Scaling ensures that no single feature dominates the model training due to its larger numerical range.
    * In this project, `StandardScaler` was applied to all features (`X`) before PCA, transforming them to have a mean of 0 and a standard deviation of 1. This is essential for PCA, which is sensitive to feature scales.
* **Dimensionality Reduction (PCA)**:
    * PCA (Principal Component Analysis) was applied to the scaled features.
    * `pca = PCA(n_components=0.70)`: This line indicates that PCA was configured to retain components that explain 70% of the total variance in the data. This effectively reduces the dimensionality of the dataset while preserving a significant portion of its information, potentially speeding up training and reducing noise. The output `X_pca` represents the data in this reduced, principal component space.

## 3. Exploratory Data Analysis (EDA)

EDA helps in understanding the underlying patterns and relationships within the data.

* **Visualize Class Distribution**: The `sns.countplot(x='Class', data=df)` and `plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')` lines in the code generate the "Class Distribution" bar plot. This visually confirms the severe class imbalance, highlighting the disparity between non-fraudulent and fraudulent transactions.
* **Feature Correlation Heatmap**:
    * `sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f")` creates a heatmap of the feature correlations.
    * The "Feature Correlation Heatmap" visually represents the correlation matrix. Strong positive correlations (red) and strong negative correlations (blue) are immediately apparent. The diagonal is always 1 (a feature is perfectly correlated with itself).
    * This heatmap is crucial for understanding how features relate to each other and to the `Class` variable. For instance, observing correlations between `Class` and other `V` features can give insights into which features are most indicative of fraud. Strong correlations (either positive or negative) with the `Class` variable would suggest features highly relevant for fraud detection.

## 4. Handle Class Imbalance

Addressing class imbalance is critical for building an effective fraud detection model.

* **Strategy Chosen: Oversampling (SMOTE)**
    * The code uses `SMOTE (Synthetic Minority Over-sampling Technique)` to handle the imbalance.
    * `smote = SMOTE(random_state=42, sampling_strategy=0.2)`: SMOTE generates synthetic samples for the minority class (fraudulent transactions) based on the existing minority class samples.
    * `sampling_strategy=0.2` means that the number of minority class samples after oversampling will be 20% of the number of majority class samples. This helps to reduce the imbalance without making the minority class as large as the majority class, which can sometimes lead to overfitting.
    * `X_pca, y = smote.fit_resample(X_pca, y)`: This line applies SMOTE to the PCA-transformed data, resulting in a more balanced dataset for training.
    * The `y_train.value_counts()` output after SMOTE (from the output of the code execution) shows the new distribution: `0` (non-fraud) around `200000` and `1` (fraud) around `40000`. This demonstrates the successful oversampling of the minority class.

## 5. Model Selection

For this project, a **Random Forest Classifier** was chosen.

* **Random Forest Classifier**:
    * `rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_depth=15, max_features='sqrt')`
    * **Ensemble Method**: Random Forests are ensemble learning methods that construct a multitude of decision trees during training and output the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
    * **Robustness**: They are known for their robustness to overfitting and ability to handle high-dimensional data.
    * **Parameters**:
        * `n_estimators=100`: The number of trees in the forest. More trees generally lead to better performance but also increase computation time.
        * `n_jobs=-1`: Uses all available processors for parallel processing, speeding up training.
        * `random_state=42`: Ensures reproducibility of the results.
        * `max_depth=15`: Limits the maximum depth of each tree, helping to prevent overfitting.
        * `max_features='sqrt'`: The number of features to consider when looking for the best split is the square root of the total number of features. This introduces randomness and helps reduce correlation between trees.

## 6. Model Evaluation

Traditional accuracy is misleading in imbalanced datasets. Therefore, other metrics are crucial.

* **Confusion Matrix**:
    ```
    [[85333    31]
     [  479 16511]]
    ```
    * **True Negatives (TN)**: 85333 (correctly predicted non-fraudulent transactions)
    * **False Positives (FP)**: 31 (non-fraudulent transactions incorrectly predicted as fraudulent – Type I error)
    * **False Negatives (FN)**: 479 (fraudulent transactions incorrectly predicted as non-fraudulent – Type II error)
    * **True Positives (TP)**: 16511 (correctly predicted fraudulent transactions)
    * The goal in fraud detection is often to minimize False Negatives, as failing to detect fraud can be costly.

* **Classification Report**:
    ```
                      precision    recall  f1-score   support

                   0       0.99      1.00      1.00     85364
                   1       1.00      0.97      0.98     16990

            accuracy                           1.00    102354
           macro avg       1.00      0.99      0.99    102354
        weighted avg       1.00      1.00      0.99    102354
    ```
    * **Precision**: For class 1 (fraud), precision is 1.00. This means that out of all transactions predicted as fraud, 100% were actually fraud. This indicates a very low rate of false positives, which is good for minimizing inconvenience to legitimate users.
    * **Recall (Sensitivity)**: For class 1 (fraud), recall is 0.97. This means the model correctly identified 97% of all actual fraudulent transactions. This is critical in fraud detection as it minimizes missed fraud.
    * **F1-Score**: The F1-score is the harmonic mean of precision and recall. For class 1, it's 0.98, indicating a good balance between precision and recall.
    * **Support**: The number of actual occurrences of the class in the specified dataset.

* **Precision-Recall Curve and AUC-PR**:
    * The Precision-Recall (PR) curve plots precision against recall for various threshold settings.
    * **Area Under the Precision-Recall Curve (AUC-PR)**: The AUC-PR is a single number summarizing the information in the PR curve. A higher AUC-PR indicates better performance, especially for imbalanced datasets. The plot shows an **AUC-PR = 0.98**, which is an excellent score, demonstrating the model's strong ability to balance precision and recall for the minority class.

## 7. Results

The trained Random Forest Classifier achieved impressive results:

* **Accuracy Training (on test set)**: 0.995017292924556 (approximately 99.5%)
* **Confusion Matrix**:
    * Very few False Positives (31), meaning legitimate transactions are rarely flagged as fraudulent.
    * Relatively low False Negatives (479), indicating that a high percentage of actual fraudulent transactions are detected.
* **Classification Report Highlights for Fraud (Class 1)**:
    * **Precision**: 1.00 (Excellent, almost no false alarms for fraud)
    * **Recall**: 0.97 (Very good, 97% of actual frauds were caught)
    * **F1-Score**: 0.98 (Strong balance)
* **AUC-PR**: 0.98 (Demonstrates robust performance in identifying fraud given the class imbalance).

These metrics collectively indicate that the model performs exceptionally well in identifying fraudulent transactions while maintaining a low rate of false alarms, which is crucial for practical fraud detection systems.

## 8. How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install necessary libraries:**
    ```bash
    pip install numpy matplotlib pandas scikit-learn seaborn imblearn
    ```
3.  **Download the dataset:**
    Place the `creditcard.csv` file in the root directory of the project. This dataset is typically available on platforms like Kaggle (e.g., "Credit Card Fraud Detection" dataset).
4.  **Run the Jupyter Notebook or Python script:**
    If the code is in a Jupyter Notebook, open it with `jupyter notebook` and run all cells. If it's a Python script, execute it using `python your_script_name.py`.

## 9. Future Enhancements

* **Hyperparameter Tuning**: More extensive hyperparameter tuning (e.g., using GridSearchCV or RandomizedSearchCV) could further optimize the Random Forest model or explore other algorithms.
* **Advanced Imbalance Handling**: Experiment with other oversampling techniques (e.g., ADASYN) or undersampling techniques (e.g., NearMiss) or a combination of both.
* **Anomaly Detection Algorithms**: Explore unsupervised anomaly detection algorithms like Isolation Forest or One-Class SVM, which are specifically designed for outlier detection and could be beneficial for fraud detection where fraud patterns might evolve over time.
* **Deep Learning Models**: Investigate neural networks (e.g., LSTMs for sequential data if time series features are explored, or simple Feed-forward networks) for potentially better performance on complex patterns.
* **Feature Engineering**: Create new features from existing ones (e.g., velocity features like transactions per unit time, or ratio of amount to average amount) to potentially provide more indicative signals to the model.
* **Cross-Validation**: Implement stratified k-fold cross-validation to ensure the model's performance is consistent across different subsets of the data, especially important given the class imbalance.