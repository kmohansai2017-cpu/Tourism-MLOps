"""
prep.py
Data Preparation Script:
1. Loads the dataset from the repository data folder
2. Removes unnecessary columns (CustomerID)
3. Handles missing values
4. Splits the data into training and testing sets
5. Saves them locally as CSV files
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Path to the dataset
DATA_PATH = "tourism_project/data/tourism.csv"

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"Dataset loaded: {df.shape}")

    # Drop unnecessary columns
    df.drop(columns=["CustomerID"], inplace=True)
    print("Dropped 'CustomerID' column.")

    # Handle missing values
    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in '{col}' with median.")

    # Categorical columns: fill with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"Filled missing values in '{col}' with mode.")

    print(f"After cleaning: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")

    # Separate features and target
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Save splits as CSV files
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print("\nData splits saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
