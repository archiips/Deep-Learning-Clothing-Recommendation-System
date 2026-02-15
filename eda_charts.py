import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style to look professional
sns.set_theme(style="whitegrid")

# Load the cleaned data (using the one with all rows for better business context)
df = pd.read_csv('dataset/Cleaned_Recommendation_Data.csv')

# Create a folder for your charts if it doesn't exist
if not os.path.exists('charts'):
    os.makedirs('charts')

# 1. Univariate Analysis: Rating Distribution (Class Imbalance)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Rating', hue='Rating', palette='viridis', legend=False)
plt.title('Univariate Analysis: Frequency of Ratings')
plt.xlabel('Stars')
plt.ylabel('Count')
plt.savefig('charts/rating_distribution.png')
print("Saved: rating_distribution.png")

# 2. Bivariate Analysis: Ratings by Division (Categorical-Numerical)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Division.Name', y='Rating', palette='magma')
plt.title('Bivariate Analysis: Rating Distribution by Division')
plt.savefig('charts/ratings_by_division.png')
print("Saved: ratings_by_division.png")

# 3. Bivariate Analysis: Recommendation vs Rating (Contingency Analysis)
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Rating', y='Recommended.IND', palette='coolwarm')
plt.title('Bivariate Analysis: Probability of Recommendation by Rating')
plt.ylabel('Recommendation Rate (0 to 1)')
plt.savefig('charts/recommend_probability.png')
print("Saved: recommend_probability.png")

print("\nEDA Phase 3 Complete. Check the 'charts' folder for your images.")
