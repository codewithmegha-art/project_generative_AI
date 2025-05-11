import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
file_path = 'HousingPrices_New.csv'
df = pd.read_csv(file_path)

# Step 2: Initial Data Exploration
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# Step 3: Data Cleaning

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

# Fill missing numerical values with the median of the column
df = df.fillna(df.median(numeric_only=True))

# Fill missing categorical values with the mode (most frequent value)
df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x)

# Step 4: Data Preparation

# Encoding categorical variables (if any)
if df.select_dtypes(include='object').shape[1] > 0:
    df = pd.get_dummies(df, drop_first=True)

# Step 5: Data Analysis

# Correlation matrix
print("\nCorrelation matrix:")
print(df.corr())

# Step 6: Data Visualization

# Correlation heatmap with updated configurations
plt.figure(figsize=(20, 16))  # Increase the size
sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    annot_kws={"size": 8}
)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# Distribution of target variable (e.g., SalePrice)
if 'SalePrice' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df['SalePrice'], kde=True)
    plt.title('Sale Price Distribution')
    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
    plt.show()

# Pairplot of numeric columns (limit to first 5 numerical columns)
sns.pairplot(df.select_dtypes(include='number').iloc[:, :5])  # limit to first 5 numerical cols
plt.show()
