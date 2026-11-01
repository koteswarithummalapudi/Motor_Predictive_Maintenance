import pandas as pd

# 1. Load the dataset
data = pd.read_csv("predicta_dataset_v2.csv")

# 2. Display dataset size
print("DATASET INFORMATION")
print("--------------------")
print("Total rows:", data.shape[0])
print("Total columns:", data.shape[1])

# 3. Display column names
print("\nColumn names:")
print(data.columns.tolist())

# 4. Display first five records
print("\nFirst five records:")
print(data.head())

# 5. Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# 6. Count each motor health category
print("\nMotor health category counts:")
print(data["Health"].value_counts())

# 7. Display numerical statistics
print("\nNumerical data summary:")
print(data[["Vibration", "Temperature", "Current"]].describe())
