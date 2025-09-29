# Heart Failure Prediction Project

Author
Zofia Seweryńska
Project completed for SKN Data Science

A comprehensive machine learning project for predicting heart disease risk using clinical patient data. This project analyzes 11 key clinical features to identify patients at risk of cardiovascular disease, achieving 85.9% accuracy with logistic regression.

## Dataset

**Source**: [Heart Failure Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

**Features** (918 patients):
- Age, Sex, Chest Pain Type
- Resting Blood Pressure & ECG Results
- Cholesterol Levels, Fasting Blood Sugar
- Maximum Heart Rate, Exercise-Induced Angina
- ST Depression (Oldpeak), ST Slope

---

## Project Workflow

### 1. Data Preprocessing
- Handled 172 missing cholesterol values using median imputation
- One-hot encoding for categorical variables (ChestPainType, ST_Slope, RestingECG)
- Z-score standardization for continuous features
- Binary mapping for Sex and ExerciseAngina

### 2. Exploratory Data Analysis
- Correlation heatmap analysis
- Distribution analysis with histograms
- Boxplots comparing features across heart disease groups
- Identified key risk factors: ST_Slope, Oldpeak, ExerciseAngina, Sex

### 3. Feature Selection
Top predictors identified:
- Sex (strongest predictor)
- Chest Pain Type (TA, ATA, NAP)
- ST Slope (Flat, Up)
- Age, MaxHR, Oldpeak, ExerciseAngina

### 4. Model Training & Evaluation
Tested three classification algorithms with GridSearchCV:
- **Logistic Regression**: 85.87% test accuracy ✓ Best Model
- Random Forest: 85.14% test accuracy
- Support Vector Machine: 85.51% test accuracy

**Performance Metrics**:
- Training Score: 85.97%
- Validation Score: 86.53%
- Test Score: 85.87%

---

## Key Findings

- **Best Model**: Logistic Regression with L1 regularization (C=1.0)
- **Most Important Features**: Sex, ChestPainType_TA, ST_Slope_Flat
- **Clinical Validity**: Results align with established cardiovascular risk factors
- **Strong Generalization**: Consistent performance across train/validation/test splits

---

## Tools & Libraries

**Core Libraries**:
- Python
- Pandas, NumPy – Data manipulation
- Scikit-learn – Machine learning models
- Seaborn, Matplotlib – Visualization

**Models Used**:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

** Future steps:
Implement SMOTE for handling class imbalance
Feature engineering for improved predictions
Cross validation usage

