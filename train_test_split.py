"""
Train/Test Split - Phase 3.4
User-based split to prevent data leakage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# Load data with user IDs
print("Loading data with user IDs...")
df = pd.read_csv('dataset/RecSys_Training_Data_with_UserIDs.csv')
print(f"Total records: {len(df)}")
print(f"Total users: {df['User.ID'].nunique()}")
print(f"Total items: {df['Clothing.ID'].nunique()}")

# ============================================================================
# Task 3.4.1: Design Split Strategy
# ============================================================================
print("\n=== DESIGNING SPLIT STRATEGY ===")

# User-based split: no user appears in both train and test
# This simulates making recommendations for new users
print("Using user-based split (80/20)")
print("No user will appear in both training and test sets")

# Get unique users with their statistics
user_stats = df.groupby('User.ID').agg({
    'Clothing.ID': 'count',  # number of reviews
    'Department.Name': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # favorite department
}).reset_index()
user_stats.columns = ['User.ID', 'num_reviews', 'favorite_department']

print(f"\nUser statistics:")
print(user_stats['num_reviews'].describe())

# ============================================================================
# Task 3.4.2: Implement Split
# ============================================================================
print("\n=== IMPLEMENTING SPLIT ===")

# Stratify by department to ensure representation
# We'll stratify by users' favorite department
train_users, test_users = train_test_split(
    user_stats['User.ID'].values,
    test_size=0.2,
    random_state=42,
    stratify=user_stats['favorite_department'].values
)

print(f"Train users: {len(train_users)}")
print(f"Test users: {len(test_users)}")

# Split the data
train_df = df[df['User.ID'].isin(train_users)]
test_df = df[df['User.ID'].isin(test_users)]

print(f"\nTrain interactions: {len(train_df)}")
print(f"Test interactions: {len(test_df)}")

# Verify no overlap
assert len(set(train_users) & set(test_users)) == 0, "User overlap detected!"
print("✓ No user overlap between train and test")

# Check department distribution
print("\n=== VERIFYING STRATIFICATION ===")
print("\nDepartment distribution in train set:")
print(train_df['Department.Name'].value_counts(normalize=True).sort_index())
print("\nDepartment distribution in test set:")
print(test_df['Department.Name'].value_counts(normalize=True).sort_index())

# Check item coverage
train_items = set(train_df['Clothing.ID'].unique())
test_items = set(test_df['Clothing.ID'].unique())
common_items = train_items & test_items
print(f"\nItems in train set: {len(train_items)}")
print(f"Items in test set: {len(test_items)}")
print(f"Items in both sets: {len(common_items)} ({len(common_items)/len(test_items):.1%} of test items)")

# Items only in test (cold-start items)
cold_start_items = test_items - train_items
print(f"Cold-start items (only in test): {len(cold_start_items)}")

# ============================================================================
# Save Split Data
# ============================================================================
print("\n=== SAVING SPLIT DATA ===")

# Save train and test sets
train_df.to_csv('dataset/train_set.csv', index=False)
test_df.to_csv('dataset/test_set.csv', index=False)
print("Saved: dataset/train_set.csv")
print("Saved: dataset/test_set.csv")

# Save user ID lists for reproducibility
with open('dataset/train_user_ids.pkl', 'wb') as f:
    pickle.dump(train_users, f)
with open('dataset/test_user_ids.pkl', 'wb') as f:
    pickle.dump(test_users, f)
print("Saved: dataset/train_user_ids.pkl")
print("Saved: dataset/test_user_ids.pkl")

# ============================================================================
# Document Split Statistics
# ============================================================================
print("\n=== SPLIT STATISTICS ===")

stats = {
    'Train Set': {
        'Users': len(train_users),
        'Items': len(train_items),
        'Interactions': len(train_df),
        'Avg interactions/user': len(train_df) / len(train_users),
        'Avg interactions/item': len(train_df) / len(train_items),
        'Sparsity': 1 - (len(train_df) / (len(train_users) * len(train_items)))
    },
    'Test Set': {
        'Users': len(test_users),
        'Items': len(test_items),
        'Interactions': len(test_df),
        'Avg interactions/user': len(test_df) / len(test_users),
        'Avg interactions/item': len(test_df) / len(test_items),
        'Sparsity': 1 - (len(test_df) / (len(test_users) * len(test_items)))
    }
}

for split_name, split_stats in stats.items():
    print(f"\n{split_name}:")
    for metric, value in split_stats.items():
        if isinstance(value, float):
            if 'Sparsity' in metric:
                print(f"  {metric}: {value:.4%}")
            else:
                print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:,}")

# Save statistics
stats_df = pd.DataFrame(stats).T
stats_df.to_csv('dataset/split_statistics.csv')
print("\nSaved: dataset/split_statistics.csv")

print("\n✅ Train/test split complete!")
