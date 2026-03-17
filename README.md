# Credit Card Fraud Detection
An end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The dataset is highly imbalanced where only 0.17% of transactions are fraud which makes this a realistic and technically challenging problem. The project covers exploratory analysis, preprocessing, class imbalance handling, model training and comparison, threshold tuning, and a deployed interactive dashboard.
The core objective is to maximize fraud detection (recall) while keeping false alarms at an acceptable level, which reflects how fraud detection systems are actually evaluated in production environments.
Live dashboard: https://creditfrauddetectionn.streamlit.app/

# The Problem
A model that predicts every transaction as legitimate achieves 99.83% accuracy on this dataset while catching exactly zero fraud cases. That number looks great on paper and is completely useless in practice. This project focuses on the metrics that actually matter in fraud detection like recall, precision, and the tradeoff between them and demonstrates how threshold tuning and resampling decisions affect real-world deployability.
Total transactions: 284,807
Fraud cases: 492 (0.17%)
Missing values: None
Features: 30 input features + 1 target

# Dataset
Source: Kaggle — Credit Card Fraud Detection
The dataset contains 284,807 transactions made by European cardholders over two days in September 2013. Features V1 through V28 are the result of a PCA transformation applied to protect cardholder identity the original features are not available. Time and Amount are the only untransformed features. The target column Class is binary: 0 for legitimate, 1 for fraud.

# Results
1. Isolation Forest: Fraud Precision- 0.31, Fraud Recall-0.33, F1 Score- 0.32, AUC-ROC- N/A
2. Random Forest: Fraud Precision-0.81, Fraud Recall- 0.81, F1 Score- 0.81, AUC-ROC-0.9688
3. XGBoost (Tuned): Fraud Precision-0.62 , Fraud Recall- 0.85 , F1 Score- 0.72 , AUC-ROC- 0.976
   The tuned XGBoost model correctly identified 83 out of 98 fraud cases in the test set, missing only 15. At this detection rate and scale, the false positive volume remains manageable which is the practical threshold for a deployable fraud system.

# Methodology
1.Exploratory Data Analysis
The initial analysis confirmed the severe class imbalance with fraud accounting for just 0.17% of all transactions. Transaction amount and time distributions were examined across both classes to identify behavioral differences. No missing values were found across any of the 31 columns.
2. Preprocessing
Amount and Time were scaled using StandardScaler since V1 through V28 are already PCA-transformed and on a comparable scale. The data was split into training and test sets using an 80/20 stratified split before any resampling was applied. SMOTE was then applied exclusively on the training set to avoid data leakage, bringing the fraud class from 394 to 227,451 samples to match the normal class.
3. Modeling
Three models were trained to cover both supervised and unsupervised approaches. Isolation Forest was included as an unsupervised baseline since it requires no labels during training. Random Forest served as the supervised baseline. XGBoost was selected as the primary model based on its AUC-ROC and higher fraud recall.
4. Threshold Tuning
The default 0.5 decision threshold on XGBoost produced a precision of just 0.35, meaning too many legitimate transactions were being flagged. A threshold sweep from 0.6 to 0.8 was performed to find the best precision-recall balance. A threshold of 0.8 was selected, improving precision to 0.62 while maintaining recall at 0.85. This single tuning step had more impact on usable model performance than hyperparameter tuning.

# Tech Stack
1. Pandas, Numpy- Data manipulation and analysis
2. scikit-learn- preprocessing, random forest, isolation forest, evaluation metrics
3. XGBoost- primary classification model
4. visualization- matplotlib and seaborn
5. Imbalanced learn- SMOTE oversampling
6. streamlit- interactive dashboard
7. google colab- development enviroment

# Key Learnings
Accuracy is not a useful metric on imbalanced datasets. A model predicting everything as normal achieves 99.83% accuracy while detecting zero fraud which illustrates exactly why precision, recall, and AUC-ROC matter more here.
Applying SMOTE after the train/test split is critical. Resampling before splitting causes data leakage where synthetic samples from the training distribution bleed into test set evaluation and inflate reported performance in a way that does not hold up in production.
Threshold tuning had a more significant impact than hyperparameter tuning. Moving the decision threshold from 0.5 to 0.8 nearly doubled precision with minimal loss in recall a tradeoff that makes the model genuinely usable rather than theoretically strong.
Isolation Forest performed poorly compared to supervised models, which was expected. Its inclusion demonstrates the practical value of labeled data in anomaly detection and gives an honest baseline for comparison.

# Author
aadhya-1803 - github.com/aadhya-1803






