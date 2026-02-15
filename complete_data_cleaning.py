"""
Complete Data Cleaning Pipeline - Phase 3.1
Handles remaining cleaning tasks: duplicates, data quality issues
"""

import pandas as pd
import numpy as np

# Load the cleaned data
print("Loading cleaned data...")
df = pd.read_csv('dataset/Cleaned_Recommendation_Data.csv')
print(f"Initial shape: {df.shape}")

# ============================================================================
# Task 3.1.2: Handle Duplicates
# ============================================================================
print("\n=== HANDLING DUPLICATES ===")

# Check for duplicate combinations of (Age, Clothing.ID, Rating, Review.Text)
print("Checking for duplicate reviews...")
duplicates = df.duplicated(subset=['Age', 'Clothing.ID', 'Rating', 'Review.Text'], keep='first')
num_duplicates = duplicates.sum()
print(f"Found {num_duplicates} duplicate reviews")

if num_duplicates > 0:
    df = df[~duplicates]
    print(f"Removed {num_duplicates} duplicates, keeping first occurrence")

# Check for same review text appearing multiple times
print("\nChecking for duplicate review texts...")
review_duplicates = df['Review.Text'].duplicated(keep='first')
num_review_duplicates = review_duplicates.sum()
print(f"Found {num_review_duplicates} duplicate review texts")

# Check for inconsistent product metadata
print("\nChecking for inconsistent product metadata...")
product_metadata = df.groupby('Clothing.ID').agg({
    'Division.Name': lambda x: x.nunique(),
    'Department.Name': lambda x: x.nunique(),
    'Class.Name': lambda x: x.nunique()
})
inconsistent_products = product_metadata[
    (product_metadata['Division.Name'] > 1) |
    (product_metadata['Department.Name'] > 1) |
    (product_metadata['Class.Name'] > 1)
]
print(f"Found {len(inconsistent_products)} products with inconsistent metadata")
if len(inconsistent_products) > 0:
    print("Inconsistent products:")
    print(inconsistent_products)

# ============================================================================
# Task 3.1.3: Fix Data Quality Issues
# ============================================================================
print("\n=== FIXING DATA QUALITY ISSUES ===")

# Standardize capitalization: title-case all category names
print("Standardizing capitalization for category names...")
df['Division.Name'] = df['Division.Name'].str.title()
df['Department.Name'] = df['Department.Name'].str.title()
df['Class.Name'] = df['Class.Name'].str.title()

# Strip extra whitespace from all string columns
print("Stripping whitespace from string columns...")
string_columns = df.select_dtypes(include=['object']).columns
for col in string_columns:
    df[col] = df[col].str.strip()

# Outlier detection - Age
print("\nDetecting age outliers...")
age_outliers = df[(df['Age'] < 18) | (df['Age'] > 100)]
print(f"Found {len(age_outliers)} age outliers (<18 or >100)")
if len(age_outliers) > 0:
    print("Age range:", age_outliers['Age'].min(), "-", age_outliers['Age'].max())
    # Cap ages at reasonable bounds
    df.loc[df['Age'] < 18, 'Age'] = 18
    df.loc[df['Age'] > 100, 'Age'] = 100
    print("Capped ages to range [18, 100]")

# Outlier detection - Rating
print("\nValidating ratings...")
invalid_ratings = df[~df['Rating'].isin([1, 2, 3, 4, 5])]
print(f"Found {len(invalid_ratings)} invalid ratings")
if len(invalid_ratings) > 0:
    print("Invalid ratings:", invalid_ratings['Rating'].unique())

# Outlier detection - Positive Feedback Count
print("\nAnalyzing positive feedback outliers...")
viral_reviews = df[df['Positive.Feedback.Count'] >= 100]
print(f"Found {len(viral_reviews)} viral reviews (100+ feedback)")
if len(viral_reviews) > 0:
    print(f"Max feedback count: {df['Positive.Feedback.Count'].max()}")
    print("Keeping viral reviews (they are legitimate)")

# ============================================================================
# Save Fully Cleaned Data
# ============================================================================
print(f"\n=== SAVING FULLY CLEANED DATA ===")
print(f"Final shape: {df.shape}")

# Save fully cleaned version
df.to_csv('dataset/Cleaned_Recommendation_Data.csv', index=False)
print("Saved: dataset/Cleaned_Recommendation_Data.csv")

# Also update training data with same cleaning
print("\nApplying same cleaning to training data...")
df_training = pd.read_csv('dataset/RecSys_Training_Data.csv')
original_training_size = len(df_training)

# Apply same cleaning steps to training data
df_training = df_training[~df_training.duplicated(subset=['Age', 'Clothing.ID', 'Rating', 'Review.Text'], keep='first')]
df_training['Division.Name'] = df_training['Division.Name'].str.title()
df_training['Department.Name'] = df_training['Department.Name'].str.title()
df_training['Class.Name'] = df_training['Class.Name'].str.title()
string_columns_training = df_training.select_dtypes(include=['object']).columns
for col in string_columns_training:
    df_training[col] = df_training[col].str.strip()
df_training.loc[df_training['Age'] < 18, 'Age'] = 18
df_training.loc[df_training['Age'] > 100, 'Age'] = 100

df_training.to_csv('dataset/RecSys_Training_Data.csv', index=False)
print(f"Training data: {original_training_size} -> {len(df_training)} rows")
print("Saved: dataset/RecSys_Training_Data.csv")

print("\n✅ Data cleaning complete!")
print(f"Total cleaned records: {len(df)}")
print(f"Total training records: {len(df_training)}")
