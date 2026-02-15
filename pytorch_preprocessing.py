"""
PyTorch Preprocessing - Phase 3.5
Create user-item matrices, encoders, negative sampling, PyTorch datasets
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
import os

# ============================================================================
# Task 3.5.1: Create User-Item Matrix
# ============================================================================
print("=== CREATING USER-ITEM MATRICES ===")

# Load train and test data
train_df = pd.read_csv('dataset/train_set.csv')
test_df = pd.read_csv('dataset/test_set.csv')

print(f"Train set: {len(train_df)} interactions")
print(f"Test set: {len(test_df)} interactions")

# Load existing mappings
with open('dataset/user_to_idx.pkl', 'rb') as f:
    user_to_idx = pickle.load(f)
with open('dataset/item_to_idx.pkl', 'rb') as f:
    item_to_idx = pickle.load(f)

# Create train matrix
train_rows = train_df['User.ID'].map(user_to_idx).values
train_cols = train_df['Clothing.ID'].map(item_to_idx).values
train_data = train_df['Rating'].values

train_matrix = csr_matrix(
    (train_data, (train_rows, train_cols)),
    shape=(len(user_to_idx), len(item_to_idx))
)

# Create test matrix
test_rows = test_df['User.ID'].map(user_to_idx).values
test_cols = test_df['Clothing.ID'].map(item_to_idx).values
test_data = test_df['Rating'].values

test_matrix = csr_matrix(
    (test_data, (test_rows, test_cols)),
    shape=(len(user_to_idx), len(item_to_idx))
)

# Calculate and document sparsity
train_sparsity = 1 - (len(train_df) / (len(user_to_idx) * len(item_to_idx)))
test_sparsity = 1 - (len(test_df) / (len(user_to_idx) * len(item_to_idx)))

print(f"\nTrain matrix shape: {train_matrix.shape}")
print(f"Train sparsity: {train_sparsity:.4%}")
print(f"Test matrix shape: {test_matrix.shape}")
print(f"Test sparsity: {test_sparsity:.4%}")

# Save matrices
save_npz('dataset/train_matrix.npz', train_matrix)
save_npz('dataset/test_matrix.npz', test_matrix)
print("\nSaved sparse matrices")

# ============================================================================
# Task 3.5.2: Integer Encoding for Embeddings
# ============================================================================
print("\n=== INTEGER ENCODING FOR EMBEDDINGS ===")

# The user_to_idx and item_to_idx are already created, but we need to encode categories
# Load full dataset to get all categories
full_df = pd.concat([train_df, test_df])

# Encode Division.Name
division_encoder = LabelEncoder()
full_df['Division.Encoded'] = division_encoder.fit_transform(full_df['Division.Name'])
train_df['Division.Encoded'] = division_encoder.transform(train_df['Division.Name'])
test_df['Division.Encoded'] = division_encoder.transform(test_df['Division.Name'])

print(f"Division classes: {len(division_encoder.classes_)}")
print(f"Division mapping: {dict(zip(division_encoder.classes_, range(len(division_encoder.classes_))))}")

# Encode Department.Name
department_encoder = LabelEncoder()
full_df['Department.Encoded'] = department_encoder.fit_transform(full_df['Department.Name'])
train_df['Department.Encoded'] = department_encoder.transform(train_df['Department.Name'])
test_df['Department.Encoded'] = department_encoder.transform(test_df['Department.Name'])

print(f"\nDepartment classes: {len(department_encoder.classes_)}")
print(f"Department mapping: {dict(zip(department_encoder.classes_, range(len(department_encoder.classes_))))}")

# Encode Class.Name
class_encoder = LabelEncoder()
full_df['Class.Encoded'] = class_encoder.fit_transform(full_df['Class.Name'])
train_df['Class.Encoded'] = class_encoder.transform(train_df['Class.Name'])
test_df['Class.Encoded'] = class_encoder.transform(test_df['Class.Name'])

print(f"\nClass classes: {len(class_encoder.classes_)}")
print(f"Class mapping: {dict(zip(class_encoder.classes_, range(len(class_encoder.classes_))))}")

# Save encoders
encoders = {
    'division': division_encoder,
    'department': department_encoder,
    'class': class_encoder,
    'user_to_idx': user_to_idx,
    'item_to_idx': item_to_idx,
    'idx_to_user': {v: k for k, v in user_to_idx.items()},
    'idx_to_item': {v: k for k, v in item_to_idx.items()}
}

with open('dataset/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("\nSaved all encoders to dataset/encoders.pkl")

# Save updated train/test sets with encoded columns
train_df.to_csv('dataset/train_set_encoded.csv', index=False)
test_df.to_csv('dataset/test_set_encoded.csv', index=False)
print("Saved encoded train and test sets")

# ============================================================================
# Task 3.5.3: Negative Sampling
# ============================================================================
print("\n=== NEGATIVE SAMPLING ===")

def create_negative_samples(df, user_to_idx, item_to_idx, n_negatives=4):
    """
    Create negative samples for implicit feedback.
    For each positive interaction, sample n_negatives items the user didn't interact with.
    """
    positive_samples = []
    negative_samples = []

    all_items = set(item_to_idx.keys())

    for user_id in df['User.ID'].unique():
        user_items = set(df[df['User.ID'] == user_id]['Clothing.ID'].values)
        user_idx = user_to_idx[user_id]

        # Positive samples
        for item_id in user_items:
            item_idx = item_to_idx[item_id]
            positive_samples.append({
                'user_idx': user_idx,
                'item_idx': item_idx,
                'label': 1
            })

            # Sample negative items
            negative_items = list(all_items - user_items)
            if len(negative_items) >= n_negatives:
                sampled_negatives = np.random.choice(negative_items, n_negatives, replace=False)
            else:
                sampled_negatives = np.random.choice(negative_items, n_negatives, replace=True)

            for neg_item_id in sampled_negatives:
                neg_item_idx = item_to_idx[neg_item_id]
                negative_samples.append({
                    'user_idx': user_idx,
                    'item_idx': neg_item_idx,
                    'label': 0
                })

    return positive_samples, negative_samples

print("Creating negative samples for training set (1:4 ratio)...")
train_pos, train_neg = create_negative_samples(train_df, user_to_idx, item_to_idx, n_negatives=4)

print(f"Positive samples: {len(train_pos)}")
print(f"Negative samples: {len(train_neg)}")
print(f"Ratio: 1:{len(train_neg) / len(train_pos):.1f}")

# Combine positive and negative samples
train_samples = train_pos + train_neg
train_samples_df = pd.DataFrame(train_samples)

# Shuffle
train_samples_df = train_samples_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total training samples (with negatives): {len(train_samples_df)}")
print(f"Label distribution:\n{train_samples_df['label'].value_counts()}")

# Save negative sampling data
train_samples_df.to_csv('dataset/train_samples_with_negatives.csv', index=False)
print("Saved: dataset/train_samples_with_negatives.csv")

# ============================================================================
# Task 3.5.4: Create PyTorch Dataset & DataLoader
# ============================================================================
print("\n=== CREATING PYTORCH DATASETS ===")

class RecommendationDataset(Dataset):
    """
    PyTorch Dataset for recommendation system.
    Supports both explicit (rating) and implicit (binary) feedback.
    """
    def __init__(self, data_df, user_to_idx, item_to_idx, mode='explicit'):
        """
        Args:
            data_df: DataFrame with user, item, and target columns
            user_to_idx: mapping from user ID to index
            item_to_idx: mapping from item ID to index
            mode: 'explicit' (predict rating) or 'implicit' (predict binary label)
        """
        self.mode = mode

        if mode == 'explicit':
            # For rating prediction
            self.users = data_df['User.ID'].map(user_to_idx).values
            self.items = data_df['Clothing.ID'].map(item_to_idx).values
            self.ratings = data_df['Rating'].values.astype(np.float32)
        else:
            # For implicit feedback (binary classification)
            self.users = data_df['user_idx'].values
            self.items = data_df['item_idx'].values
            self.labels = data_df['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = torch.LongTensor([self.users[idx]])
        item = torch.LongTensor([self.items[idx]])

        if self.mode == 'explicit':
            target = torch.FloatTensor([self.ratings[idx]])
        else:
            target = torch.FloatTensor([self.labels[idx]])

        return user, item, target

# Create datasets
print("Creating explicit feedback datasets (rating prediction)...")
train_dataset_explicit = RecommendationDataset(train_df, user_to_idx, item_to_idx, mode='explicit')
test_dataset_explicit = RecommendationDataset(test_df, user_to_idx, item_to_idx, mode='explicit')

print(f"Train dataset (explicit): {len(train_dataset_explicit)} samples")
print(f"Test dataset (explicit): {len(test_dataset_explicit)} samples")

print("\nCreating implicit feedback dataset (binary classification)...")
train_dataset_implicit = RecommendationDataset(train_samples_df, user_to_idx, item_to_idx, mode='implicit')

print(f"Train dataset (implicit): {len(train_dataset_implicit)} samples")

# Create DataLoaders
batch_size_mf = 256
batch_size_ncf = 512

train_loader_explicit_mf = DataLoader(
    train_dataset_explicit,
    batch_size=batch_size_mf,
    shuffle=True,
    num_workers=0
)

train_loader_explicit_ncf = DataLoader(
    train_dataset_explicit,
    batch_size=batch_size_ncf,
    shuffle=True,
    num_workers=0
)

train_loader_implicit = DataLoader(
    train_dataset_implicit,
    batch_size=batch_size_ncf,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset_explicit,
    batch_size=batch_size_ncf,
    shuffle=False,
    num_workers=0
)

print(f"\nCreated DataLoaders:")
print(f"  - Train (MF, explicit): batch_size={batch_size_mf}, {len(train_loader_explicit_mf)} batches")
print(f"  - Train (NCF, explicit): batch_size={batch_size_ncf}, {len(train_loader_explicit_ncf)} batches")
print(f"  - Train (implicit): batch_size={batch_size_ncf}, {len(train_loader_implicit)} batches")
print(f"  - Test: batch_size={batch_size_ncf}, {len(test_loader)} batches")

# Save a sample batch to verify
print("\n=== SAMPLE BATCH (Explicit) ===")
sample_users, sample_items, sample_ratings = next(iter(train_loader_explicit_mf))
print(f"User batch shape: {sample_users.shape}")
print(f"Item batch shape: {sample_items.shape}")
print(f"Rating batch shape: {sample_ratings.shape}")
print(f"Sample users: {sample_users[:5].squeeze()}")
print(f"Sample items: {sample_items[:5].squeeze()}")
print(f"Sample ratings: {sample_ratings[:5].squeeze()}")

print("\n=== SAMPLE BATCH (Implicit) ===")
sample_users, sample_items, sample_labels = next(iter(train_loader_implicit))
print(f"User batch shape: {sample_users.shape}")
print(f"Item batch shape: {sample_items.shape}")
print(f"Label batch shape: {sample_labels.shape}")
print(f"Sample labels: {sample_labels[:10].squeeze()}")

# ============================================================================
# Save Configuration
# ============================================================================
print("\n=== SAVING CONFIGURATION ===")

config = {
    'n_users': len(user_to_idx),
    'n_items': len(item_to_idx),
    'n_divisions': len(division_encoder.classes_),
    'n_departments': len(department_encoder.classes_),
    'n_classes': len(class_encoder.classes_),
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'train_samples_with_negatives': len(train_samples_df),
    'batch_size_mf': batch_size_mf,
    'batch_size_ncf': batch_size_ncf,
    'sparsity_train': float(train_sparsity),
    'sparsity_test': float(test_sparsity)
}

with open('dataset/pytorch_config.pkl', 'wb') as f:
    pickle.dump(config, f)

print("Configuration saved to dataset/pytorch_config.pkl")
print("\nConfig:")
for key, value in config.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value:,}")

print("\n✅ PyTorch preprocessing complete!")
print("\nCreated files:")
print("  - dataset/train_matrix.npz")
print("  - dataset/test_matrix.npz")
print("  - dataset/train_set_encoded.csv")
print("  - dataset/test_set_encoded.csv")
print("  - dataset/encoders.pkl")
print("  - dataset/train_samples_with_negatives.csv")
print("  - dataset/pytorch_config.pkl")
print("\nReady for Phase 4: Model Development!")
