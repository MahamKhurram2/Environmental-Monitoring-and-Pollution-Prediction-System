import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to CSV files
pollution_path = r"c:\Users\hp\OneDrive\Desktop\Semester 7\Mlops\project\course-project-mahamkhurram\task1\data\pollution.csv"
weather_path = r"c:\Users\hp\OneDrive\Desktop\Semester 7\Mlops\project\course-project-mahamkhurram\task1\data\weather.csv"

# Check if files exist
if not os.path.exists(pollution_path):
    raise FileNotFoundError(f"File not found: {pollution_path}")
if not os.path.exists(weather_path):
    raise FileNotFoundError(f"File not found: {weather_path}")

# Load CSV files
pollution_df = pd.read_csv(pollution_path)
weather_df = pd.read_csv(weather_path)

# Ensure both have a 'timestamp' column to merge on
if 'timestamp' not in pollution_df.columns or 'timestamp' not in weather_df.columns:
    raise ValueError("Both CSV files must have a 'timestamp' column to merge on.")

# Merge datasets on timestamp
merged_df = pd.merge(pollution_df, weather_df, on='timestamp', how='inner')

print("Initial Merged Data (Head):")
print(merged_df.head())

# Identify numeric columns for imputation and outlier handling
# Excluding 'timestamp' and 'city' as they are not numeric
numeric_columns = ['aqi', 'pm2_5', 'pm10', 'temperature', 'humidity', 'pressure']

# Check Missing Values
print("\nMissing Values Before Handling:")
print(merged_df.isnull().sum())

# Impute missing numeric values
for col in numeric_columns:
    if merged_df[col].notna().sum() > 0:
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    else:
        # If all values are NaN, fill with 0
        merged_df[col] = merged_df[col].fillna(0)

print("\nMissing Values After Handling:")
print(merged_df.isnull().sum())

# Check if there's numeric data to plot before outlier removal
valid_before = merged_df[numeric_columns].dropna(how='all')
if valid_before.empty or len(valid_before.columns) == 0:
    print("No numeric data available to plot before outlier removal. Skipping boxplot.")
else:
    sns.boxplot(data=valid_before)
    plt.title('Boxplot for Numeric Features Before Outlier Removal')
    plt.show()

# Remove outliers using IQR
Q1 = merged_df[numeric_columns].quantile(0.25)
Q3 = merged_df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
threshold = 1.5
condition = ~((merged_df[numeric_columns] < (Q1 - threshold * IQR)) |
              (merged_df[numeric_columns] > (Q3 + threshold * IQR))).any(axis=1)
merged_df_no_outliers = merged_df[condition]

# Check numeric data after outlier removal
valid_after = merged_df_no_outliers[numeric_columns].dropna(how='all')
if valid_after.empty or len(valid_after.columns) == 0:
    print("No numeric data available to plot after outlier removal. Skipping boxplot.")
else:
    sns.boxplot(data=valid_after)
    plt.title('Boxplot After Removing Outliers')
    plt.show()

# Ensure output directory exists
os.makedirs('data', exist_ok=True)

# Save the cleaned merged data to a CSV
cleaned_data_path = 'data/cleaned_merged_data.csv'
merged_df_no_outliers.to_csv(cleaned_data_path, index=False)
print(f"\nCleaned data saved to {cleaned_data_path}")
print("Data preprocessing completed successfully!")
