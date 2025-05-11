import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Step 1: Load the Dataset
file_path = 'HousingPrices_New.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Initial Data Exploration
print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Step 3: Data Cleaning

# Drop duplicate rows
df = df.drop_duplicates()

# Fill missing numerical values with the median of the column
df = df.fillna(df.median(numeric_only=True))

# Fill missing categorical values with the mode (most frequent value)
df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x)

# Check the cleaned dataset for missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Step 4: Data Transformation

# Encoding categorical variables using one-hot encoding
if df.select_dtypes(include='object').shape[1] > 0:
    df = pd.get_dummies(df, drop_first=True)

# Display the dataset after transformation
print("\nData after transformation (first 5 rows):")
print(df.head())

# Step 5: Data Analysis

# Correlation Matrix
correlation_matrix = df.corr()

# Print Correlation Matrix
print("\nCorrelation matrix:")
print(correlation_matrix)

# Visualize the correlation matrix with a heatmap
plt.figure(figsize=(20, 16))  # Increase the size
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# Step 6: Target Variable Distribution (e.g., SalePrice)

# Visualize the distribution of SalePrice (or target variable)
if 'SalePrice' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df['SalePrice'], kde=True)
    plt.title('Sale Price Distribution')
    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
    plt.show()

# Step 7: Feature Relationships (Pairplot)

# Pairplot of numeric features to visualize relationships between them
sns.pairplot(df.select_dtypes(include='number').iloc[:, :5])  # limit to first 5 numerical columns
plt.show()

# Step 8: Model Preparation (Example)

# Assuming 'SalePrice' is the target variable
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of training and test sets
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

