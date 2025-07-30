# This project is based on a kaggle data set of Heart Failure Prediction
# Done by Zofia Sewerynska
# 05.2025, Poland

import os
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import statistics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#---------------------------------------------------------------------------------
# kaggle data zip
# os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\zosia\.kaggle'
# os.system('kaggle datasets download -d fedesoriano/heart-failure-prediction')
# with zipfile.ZipFile("heart-failure-prediction.zip", 'r') as zip_ref:
#     zip_ref.extractall("heart_failure_data")

# Stwórz klasyfikatora, który przewiduje ryzyko choroby serca na podstawie danych pacjenta,
# takich jak wiek, BMI i ciśnienie. Analizuj dane oraz to jak jedna cecha wpływa na inną. Zobacz,
# które z cech potrzebujesz do predykcji, a których nie (nie zmieniają wyników modelu, są mniej istotne).
# Możesz użyć różnych modeli i porównać ich wyniki ze sobą. Przetestuj otrzymane wyniki modelu.

#---------------------------------------------------------------------------------


#extracting data
df = pd.read_csv("heart_failure_data/heart.csv")
fasting = df['FastingBS'].value_counts(normalize=True)  #sprawdzanie ilosci wartosci
resting = df['RestingECG'].value_counts(normalize=True)

df.duplicated().sum() #zero
df.isna().sum() #zero

#ChestPainType, ST_Slope, RestingECG - ONE HOT ENCODING
categorical_features = ["ChestPainType", "ST_Slope", "RestingECG"]
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

#numerical values
df["Sex"] = df["Sex"].map({"M": 0, "F": 1})
df["ExerciseAngina"] = df["ExerciseAngina"].map({"N": 0, "Y": 1})


#standaryzacja danych - srednia = 0 i st odchylenie = 1
# z = (x - mean) / standard_deviation
# Age, RestingBP, Cholesterol, MaxHR, Oldpeak are  continuous variables so should be standarized
# categorical features (Sex, ChestPainType, ST_Slope, ExerciseAngina, etc.) should be left numeric but not standardized

def standardize_features(columns):
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / std
    return df
columns_to_standardize = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

df = standardize_features(columns_to_standardize)
# print(df.head())

#Analiza danych:

# sns.countplot(x = "HeartDisease", data = df)
# # plt.show()
#
# plt.figure(figsize=(7,7))
# sns.heatmap(df.corr(), annot=True, cmap=sns.cubehelix_palette(as_cmap=True)) #df.corr() - prepered for the correlation; annot - gives the numbers, cmap - colors
# # plt.show()
#
# df.hist(bins=30, figsize=(15, 10))
# plt.tight_layout()
# plt.show()

#Analiza Cholesterolu
print((df["Cholesterol"] == 0).sum()) #no zero cholesterol values
print(min(df["Cholesterol"]))
print(max(df["Cholesterol"]))


#Box ploty dla numeric data:
data_for_boxplot = ["Oldpeak", "MaxHR", "Cholesterol", "Age"]
f, axes = plt.subplots(1,len(data_for_boxplot),figsize = (14,10))
for i in range(len(data_for_boxplot)):
    sns.boxplot(data=df, x="HeartDisease", y=data_for_boxplot[i],  ax=axes[i], color = 'plum')
    axes[i].set_title(f"{data_for_boxplot[i]} by Heart Disease")
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['No', 'Yes'])
# plt.show()

#Count ploty dla binary data
data_for_countplot = ["ExerciseAngina", "Sex"]
f, axes = plt.subplots(1,len(data_for_countplot),figsize = (14,10))
for i in range(len(data_for_countplot)):
    sns.countplot(data=df, x="HeartDisease", hue=data_for_countplot[i],  ax=axes[i], palette = 'flare')
    axes[i].set_title(f"{data_for_countplot[i]} by Heart Disease")
    if data_for_countplot[i] == "ExerciseAngina":
        axes[i].legend(['No', 'Yes'])
    if data_for_countplot[i] == "Sex":
        axes[i].legend(['Male', 'Female'])
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['No', 'Yes'])
# plt.show()


# Important features
selected_features = df[['ST_Slope_Flat', 'ST_Slope_Up', 'Sex', 'Age', 'MaxHR', 'Oldpeak', 'ExerciseAngina', 'ChestPainType_TA', 'ChestPainType_ATA', 'ChestPainType_NAP']]
selected_y = df['HeartDisease']

df_to_classify = selected_features
X_train, X_test, y_train, y_test = train_test_split(df_to_classify, selected_y, test_size = 0.2, random_state = 42)

# All features
df_full = df
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(df_full, selected_y, test_size = 0.2, random_state = 42)

# Simple Linear Regression

reg = LinearRegression()
reg.fit(X_train, y_train)
# regression coefficients
print('Coefficients: ', reg.coef_)
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

plt.figure(figsize = (14,10))
plt.scatter(y_train_pred,
            y_train_pred - y_train,
            color="green", s=10,
            label='Train data')

plt.scatter(y_test_pred,
            y_test_pred - y_test,
            color="blue", s=10,
            label='Test data')

plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()


