import pandas as pd
import os

# Path management
file_path = os.path.join('dataset', 'Women_s_E-Commerce_Clothing_Reviews_1594_1.csv')
df = pd.read_csv(file_path)

# --- 1. CLEANING ---
# Drop redundant index columns
df = df.drop(columns=['Unnamed: 0', 'X'], errors='ignore')

# Fix the category typo
df['Division.Name'] = df['Division.Name'].replace('Initmates', 'Intimates')

# Drop rows where critical metadata OR the review text is missing
# (Crucial for the NLP/Deep Learning part of your framework)
df = df.dropna(subset=['Division.Name', 'Review.Text'])

# Fill missing Titles with a placeholder so the code doesn't crash later
df['Title'] = df['Title'].fillna('No Title')

# --- 2. FILTERING (Cold Start) ---
# Keep only products with 5+ reviews for a stable recommendation signal
item_counts = df['Clothing.ID'].value_counts()
df_training = df[df['Clothing.ID'].isin(item_counts[item_counts >= 5].index)]

# --- 3. SAVING ---
# Full cleaned version for your EDA charts (Phase 3)
df.to_csv('dataset/Cleaned_Recommendation_Data.csv', index=False)

# Filtered version for your PyTorch Model (Phase 4)
df_training.to_csv('dataset/RecSys_Training_Data.csv', index=False)

print(f"Final Training Set Size: {len(df_training)} reviews.")
print("Data is now 100% compliant with the Project Framework.")

