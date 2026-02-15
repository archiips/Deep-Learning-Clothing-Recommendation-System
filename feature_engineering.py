"""
Feature Engineering - Phase 3.2
Create pseudo user IDs, user features, item features, and interaction features
"""

import pandas as pd
import numpy as np
import pickle
import os

# Load training data
print("Loading training data...")
df = pd.read_csv('dataset/RecSys_Training_Data.csv')
print(f"Training data shape: {df.shape}")

# ============================================================================
# Task 3.2.1: Create Pseudo User IDs
# ============================================================================
print("\n=== CREATING PSEUDO USER IDS ===")

# Strategy: Group reviews by Age and review patterns to create pseudo users
# Since we don't have actual user IDs, we'll use Age + sequential ordering
# This is a workaround - we acknowledge the limitation

# Create pseudo user ID based on Age and review order
# Assumption: reviews from same age at similar times might be from same user
df = df.sort_values(['Age', 'Clothing.ID'])

# Create groups: each unique age gets a user ID range
# We'll assume each person reviewed on average 3-5 items
user_id_counter = 0
pseudo_user_ids = []

# Group by age and create user IDs
for age in df['Age'].unique():
    age_reviews = df[df['Age'] == age]
    num_reviews = len(age_reviews)

    # Estimate number of unique users for this age
    # Assume each user writes ~4 reviews on average
    estimated_users = max(1, num_reviews // 4)

    # Distribute reviews across estimated users
    user_ids_for_age = np.repeat(range(user_id_counter, user_id_counter + estimated_users),
                                   (num_reviews // estimated_users) + 1)[:num_reviews]

    pseudo_user_ids.extend(user_ids_for_age)
    user_id_counter += estimated_users

df['User.ID'] = pseudo_user_ids

print(f"Created {df['User.ID'].nunique()} pseudo users from {len(df)} reviews")
print(f"Average reviews per user: {len(df) / df['User.ID'].nunique():.2f}")

# Verify each user has multiple reviews
user_review_counts = df['User.ID'].value_counts()
print(f"Users with 1 review: {(user_review_counts == 1).sum()}")
print(f"Users with 2+ reviews: {(user_review_counts >= 2).sum()}")
print(f"Users with 5+ reviews: {(user_review_counts >= 5).sum()}")

# ============================================================================
# Task 3.2.2: Engineer User Features
# ============================================================================
print("\n=== ENGINEERING USER FEATURES ===")

user_features = df.groupby('User.ID').agg({
    'Rating': ['count', 'mean', 'std'],
    'Recommended.IND': 'mean',
    'Positive.Feedback.Count': 'mean',
    'Department.Name': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
    'Review.Text': lambda x: x.str.split().str.len().mean()
}).reset_index()

# Flatten column names
user_features.columns = ['User.ID', 'total_reviews', 'avg_rating', 'rating_std',
                         'recommendation_rate', 'avg_positive_feedback',
                         'favorite_department', 'review_length_avg']

# Fill NaN in rating_std (users with only 1 review)
user_features['rating_std'] = user_features['rating_std'].fillna(0)

# Add department diversity
dept_diversity = df.groupby('User.ID')['Department.Name'].nunique().reset_index()
dept_diversity.columns = ['User.ID', 'department_diversity']
user_features = user_features.merge(dept_diversity, on='User.ID')

# Add age (from first review of each user)
user_age = df.groupby('User.ID')['Age'].first().reset_index()
user_features = user_features.merge(user_age, on='User.ID')

print(f"User features shape: {user_features.shape}")
print("\nUser features summary:")
print(user_features.describe())

# Save user features
user_features.to_csv('dataset/user_features.csv', index=False)
print("\nSaved: dataset/user_features.csv")

# ============================================================================
# Task 3.2.3: Engineer Item Features
# ============================================================================
print("\n=== ENGINEERING ITEM FEATURES ===")

item_features = df.groupby('Clothing.ID').agg({
    'Rating': ['mean', 'count', 'std'],
    'Recommended.IND': 'mean',
    'Positive.Feedback.Count': 'mean',
    'Division.Name': 'first',
    'Department.Name': 'first',
    'Class.Name': 'first'
}).reset_index()

# Flatten column names
item_features.columns = ['Clothing.ID', 'avg_rating', 'rating_count', 'rating_std',
                         'recommendation_rate', 'avg_positive_feedback',
                         'division', 'department', 'class_name']

# Fill NaN in rating_std
item_features['rating_std'] = item_features['rating_std'].fillna(0)

# Calculate popularity score
item_features['popularity_score'] = (
    item_features['avg_rating'] * np.log1p(item_features['rating_count'])
)

# Add review sentiment (simple: based on recommendation rate and avg rating)
item_features['review_sentiment'] = (
    0.6 * item_features['avg_rating'] / 5 +
    0.4 * item_features['recommendation_rate']
)

print(f"Item features shape: {item_features.shape}")
print("\nItem features summary:")
print(item_features.describe())

# Save item features
item_features.to_csv('dataset/item_features.csv', index=False)
print("\nSaved: dataset/item_features.csv")

# ============================================================================
# Task 3.2.4: Create Interaction Features
# ============================================================================
print("\n=== CREATING INTERACTION FEATURES ===")

# Build user-item interaction records
interactions = df[['User.ID', 'Clothing.ID', 'Rating', 'Recommended.IND']].copy()

# Compute interaction strength
interactions['interaction_strength'] = (
    0.7 * interactions['Rating'] / 5 +
    0.3 * interactions['Recommended.IND']
)

print(f"Interactions shape: {interactions.shape}")
print("\nInteraction strength distribution:")
print(interactions['interaction_strength'].describe())

# Save interactions
interactions.to_csv('dataset/interactions.csv', index=False)
print("\nSaved: dataset/interactions.csv")

# Create sparse user-item matrix
print("\n=== CREATING USER-ITEM MATRIX ===")

from scipy.sparse import csr_matrix

# Create mappings
unique_users = sorted(df['User.ID'].unique())
unique_items = sorted(df['Clothing.ID'].unique())

user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

# Save mappings
with open('dataset/user_to_idx.pkl', 'wb') as f:
    pickle.dump(user_to_idx, f)
with open('dataset/item_to_idx.pkl', 'wb') as f:
    pickle.dump(item_to_idx, f)

print(f"Saved mappings: {len(user_to_idx)} users, {len(item_to_idx)} items")

# Create sparse matrix with ratings
rows = df['User.ID'].map(user_to_idx).values
cols = df['Clothing.ID'].map(item_to_idx).values
data = df['Rating'].values

user_item_matrix = csr_matrix((data, (rows, cols)),
                               shape=(len(unique_users), len(unique_items)))

# Calculate sparsity
n_interactions = len(df)
n_possible = len(unique_users) * len(unique_items)
sparsity = 1 - (n_interactions / n_possible)

print(f"\nUser-Item Matrix shape: {user_item_matrix.shape}")
print(f"Total interactions: {n_interactions:,}")
print(f"Possible interactions: {n_possible:,}")
print(f"Sparsity: {sparsity:.4%}")

# Save sparse matrix
from scipy.sparse import save_npz
save_npz('dataset/user_item_matrix.npz', user_item_matrix)
print("Saved: dataset/user_item_matrix.npz")

# ============================================================================
# Save Enhanced Training Data with User IDs
# ============================================================================
print("\n=== SAVING ENHANCED TRAINING DATA ===")
df.to_csv('dataset/RecSys_Training_Data_with_UserIDs.csv', index=False)
print("Saved: dataset/RecSys_Training_Data_with_UserIDs.csv")

print("\n✅ Feature engineering complete!")
print(f"\nCreated files:")
print("  - dataset/user_features.csv")
print("  - dataset/item_features.csv")
print("  - dataset/interactions.csv")
print("  - dataset/user_to_idx.pkl")
print("  - dataset/item_to_idx.pkl")
print("  - dataset/user_item_matrix.npz")
print("  - dataset/RecSys_Training_Data_with_UserIDs.csv")
