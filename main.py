# This project is based on a kaggle data set of Heart Failure Prediction
# Done by Zofia Sewerynska
# 05.2025, Poland

import os
import zipfile
#---------------------------------------------------------------------------------
# kaggle data zip
# os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\zosia\.kaggle'
# os.system('kaggle datasets download -d fedesoriano/heart-failure-prediction')
# with zipfile.ZipFile("heart-failure-prediction.zip", 'r') as zip_ref:
#     zip_ref.extractall("heart_failure_data")
#---------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import statistics

#extracting data
df = pd.read_csv("heart_failure_data/heart.csv")
fasting = df['FastingBS'].value_counts(normalize=True)  #sprawdzanie ilosci wartosci
resting = df['RestingECG'].value_counts(normalize=True)

df.duplicated().sum() #zero
df.isna().sum() #zero

#numerical values
df["Sex"] = df["Sex"].map({"M": 0, "F": 1})
df["ChestPainType"] = df["ChestPainType"].map({"ATA": 0, "NAP": 0.5, "ASY": 1})
df["RestingECG"] = df["RestingECG"].map({"Normal": 0, "LVH": 0.5, "ST": 1})
df["ST_Slope"] = df["ST_Slope"].map({"Up": 0, "Flat": 0.5, "Down": 1})
df["ExerciseAngina"] = df["ExerciseAngina"].map({"N": 0, "Y": 1})

#standaryzacja danych - srednia = 0 i st odchylenei = 1
# z = (x - mean) / standard_deviation

def standardize_features(columns):
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / std
    return df
columns_to_standardize = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR"]

df = standardize_features(columns_to_standardize)
print(df.head())




