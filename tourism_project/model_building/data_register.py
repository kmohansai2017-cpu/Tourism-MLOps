"""
data_register.py
Registers the tourism dataset by:
1. Reading the CSV from the repository data folder
2. Checking that all expected columns are present
3. Printing a short summary
"""

import pandas as pd
import sys
import os

# Path to the dataset in the repository
DATA_PATH = "tourism_project/data/tourism.csv"

# Expected columns in the dataset
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

def main():
    # Check file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    # Load the dataset
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"Dataset loaded successfully from: {DATA_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    # Verify expected columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
        sys.exit(1)
    else:
        print("All expected columns are present.\n")

    # Print summary
    print("--- Dataset Summary ---")
    print(f"Target variable (ProdTaken) distribution:")
    print(df["ProdTaken"].value_counts())
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print("\nDataset registration complete.")

if __name__ == "__main__":
    main()
