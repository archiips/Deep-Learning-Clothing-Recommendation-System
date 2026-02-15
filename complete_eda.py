"""
Complete EDA - Phase 3.3
Remaining EDA tasks: department analysis, product popularity, age demographics, review text insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# Set style
sns.set_theme(style="whitegrid")

# Load data
print("Loading cleaned data...")
df = pd.read_csv('dataset/Cleaned_Recommendation_Data.csv')

# Create charts folder
if not os.path.exists('charts'):
    os.makedirs('charts')

# ============================================================================
# Task 3.3.2: Division/Department Analysis (complete)
# ============================================================================
print("\n=== DEPARTMENT ANALYSIS ===")

# Bar chart of review counts by department
plt.figure(figsize=(12, 6))
dept_counts = df['Department.Name'].value_counts()
sns.barplot(x=dept_counts.index, y=dept_counts.values, palette='viridis')
plt.title('Review Counts by Department')
plt.xlabel('Department')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/review_counts_by_department.png')
print("Saved: charts/review_counts_by_department.png")
plt.close()

# Average rating per department
print("\nAverage rating by department:")
avg_rating_dept = df.groupby('Department.Name')['Rating'].mean().sort_values(ascending=False)
print(avg_rating_dept)

# Recommendation rate by department
print("\nRecommendation rate by department:")
rec_rate_dept = df.groupby('Department.Name')['Recommended.IND'].mean().sort_values(ascending=False)
print(rec_rate_dept)

# ============================================================================
# Task 3.3.3: Recommendation Pattern Analysis (complete)
# ============================================================================
print("\n=== RECOMMENDATION PATTERNS ===")

overall_rec_rate = df['Recommended.IND'].mean()
print(f"Overall recommendation rate: {overall_rec_rate:.2%}")

# Analyze 4-star items
four_star_recs = df[df['Rating'] == 4]['Recommended.IND'].mean()
print(f"4-star items recommendation rate: {four_star_recs:.2%}")

# ============================================================================
# Task 3.3.4: Product Popularity Analysis
# ============================================================================
print("\n=== PRODUCT POPULARITY ANALYSIS ===")

# Product popularity distribution
product_counts = df['Clothing.ID'].value_counts()

# Create log-scale histogram
plt.figure(figsize=(10, 6))
plt.hist(product_counts.values, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Reviews per Product')
plt.ylabel('Number of Products')
plt.title('Product Popularity Distribution')
plt.yscale('log')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/product_popularity_distribution.png')
print("Saved: charts/product_popularity_distribution.png")
plt.close()

# Top 10 most-reviewed items
print("\nTop 10 most-reviewed products:")
top_10 = product_counts.head(10)
print(top_10)

# Items with <5 reviews
items_below_5 = (product_counts < 5).sum()
total_items = len(product_counts)
pct_below_5 = items_below_5 / total_items
print(f"\nProducts with <5 reviews: {items_below_5} out of {total_items} ({pct_below_5:.1%})")

# 80/20 rule analysis
sorted_counts = product_counts.sort_values(ascending=False)
cumsum = sorted_counts.cumsum()
total_reviews = cumsum.iloc[-1]
top_20_pct_items = int(len(sorted_counts) * 0.2)
reviews_from_top_20 = cumsum.iloc[top_20_pct_items - 1]
pct_reviews_from_top_20 = reviews_from_top_20 / total_reviews
print(f"\n80/20 Analysis: Top 20% of items account for {pct_reviews_from_top_20:.1%} of reviews")

# ============================================================================
# Task 3.3.5: Customer Age Demographics
# ============================================================================
print("\n=== AGE DEMOGRAPHICS ===")

# Age distribution histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Customer Age Distribution')
plt.axvline(df['Age'].mean(), color='red', linestyle='--', label=f"Mean: {df['Age'].mean():.1f}")
plt.axvline(df['Age'].median(), color='green', linestyle='--', label=f"Median: {df['Age'].median():.1f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/age_distribution.png')
print("Saved: charts/age_distribution.png")
plt.close()

print(f"\nAge statistics:")
print(f"  Mean: {df['Age'].mean():.2f}")
print(f"  Median: {df['Age'].median():.2f}")
print(f"  Std: {df['Age'].std():.2f}")
print(f"  Range: {df['Age'].min()} - {df['Age'].max()}")

# Age vs rating correlation
age_rating_corr = df[['Age', 'Rating']].corr().iloc[0, 1]
print(f"\nAge vs Rating correlation: {age_rating_corr:.3f}")

# Create age groups for analysis
df['Age.Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 65, 100],
                          labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66+'])

# Average rating by age group
plt.figure(figsize=(10, 6))
age_group_ratings = df.groupby('Age.Group')['Rating'].mean()
sns.barplot(x=age_group_ratings.index, y=age_group_ratings.values, palette='coolwarm')
plt.title('Average Rating by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.savefig('charts/rating_by_age_group.png')
print("Saved: charts/rating_by_age_group.png")
plt.close()

# Recommendation rate by age group
age_group_rec = df.groupby('Age.Group')['Recommended.IND'].mean()
print(f"\nRecommendation rate by age group:")
print(age_group_rec)

# ============================================================================
# Task 3.3.6: Review Text Insights
# ============================================================================
print("\n=== REVIEW TEXT INSIGHTS ===")

# Calculate review lengths
df['Review.Length'] = df['Review.Text'].str.split().str.len()
avg_review_length = df['Review.Length'].mean()
print(f"Average review length: {avg_review_length:.1f} words")

# Review length by rating
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Rating', y='Review.Length', palette='magma')
plt.title('Review Length by Rating')
plt.xlabel('Rating')
plt.ylabel('Word Count')
plt.tight_layout()
plt.savefig('charts/review_length_by_rating.png')
print("Saved: charts/review_length_by_rating.png")
plt.close()

# Most common words in positive reviews (4-5 stars)
print("\nAnalyzing word frequencies in positive reviews...")
positive_reviews = df[df['Rating'] >= 4]['Review.Text'].str.lower()
positive_words = ' '.join(positive_reviews.astype(str)).split()
# Remove common stop words
stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'is', 'it', 'this', 'that', 'i', 'was', 'as', 'be'])
positive_words_filtered = [w for w in positive_words if w not in stop_words and len(w) > 2]
positive_word_freq = Counter(positive_words_filtered).most_common(20)
print("\nTop 20 words in positive reviews (4-5 stars):")
for word, count in positive_word_freq[:20]:
    print(f"  {word}: {count}")

# Most common words in negative reviews (1-2 stars)
print("\nAnalyzing word frequencies in negative reviews...")
negative_reviews = df[df['Rating'] <= 2]['Review.Text'].str.lower()
negative_words = ' '.join(negative_reviews.astype(str)).split()
negative_words_filtered = [w for w in negative_words if w not in stop_words and len(w) > 2]
negative_word_freq = Counter(negative_words_filtered).most_common(20)
print("\nTop 20 words in negative reviews (1-2 stars):")
for word, count in negative_word_freq[:20]:
    print(f"  {word}: {count}")

# ============================================================================
# Task 3.3.7: Cold Start & Data Sparsity Visualization
# ============================================================================
print("\n=== COLD START & DATA SPARSITY ===")

# Visualization showing review frequency distribution
plt.figure(figsize=(12, 6))
review_freq = product_counts.value_counts().sort_index()
plt.bar(review_freq.index[:50], review_freq.values[:50], alpha=0.7, edgecolor='black')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Products')
plt.title('Review Frequency Distribution (First 50 bins)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/review_frequency_distribution.png')
print("Saved: charts/review_frequency_distribution.png")
plt.close()

# Statistics
single_review_items = (product_counts == 1).sum()
pct_single_review = single_review_items / total_items
print(f"Products with only 1 review: {single_review_items} ({pct_single_review:.1%})")

items_5plus = (product_counts >= 5).sum()
pct_5plus = items_5plus / total_items
print(f"Products with 5+ reviews: {items_5plus} ({pct_5plus:.1%})")

print("\n✅ Complete EDA finished!")
print(f"Total charts created: 9")
print("Check the 'charts' folder for all visualizations")
