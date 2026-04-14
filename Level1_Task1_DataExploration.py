"""
COGNIFYZ TECHNOLOGIES - DATA SCIENCE INTERNSHIP
Level 1 | Task 1: Data Exploration and Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('Dataset .csv', encoding='utf-8-sig')

print("=" * 60)
print("   LEVEL 1 | TASK 1: DATA EXPLORATION & PREPROCESSING")
print("=" * 60)

# ── 1. Rows and Columns ────────────────────────────────────────────────────────
print(f"\n📊 Dataset Shape:")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")

print(f"\n📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")

# ── 2. Missing Values ──────────────────────────────────────────────────────────
print(f"\n🔍 Missing Values Analysis:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]

if missing_df.empty:
    print("   ✅ No missing values found except in 'Cuisines' column.")
else:
    print(missing_df)

print(f"\n   Cuisines column missing: {df['Cuisines'].isnull().sum()} rows")
# Handle missing Cuisines
df['Cuisines'].fillna('Unknown', inplace=True)
print("   ✅ Filled missing Cuisines with 'Unknown'")

# ── 3. Data Type Conversion ────────────────────────────────────────────────────
print(f"\n🔄 Data Type Conversion:")
binary_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    print(f"   ✅ '{col}' → converted to binary (0/1)")

# ── 4. Target Variable Distribution ───────────────────────────────────────────
print(f"\n🎯 Aggregate Rating Distribution:")
print(df['Aggregate rating'].describe().round(2))

# Class imbalance check
zero_ratings = (df['Aggregate rating'] == 0).sum()
non_zero = (df['Aggregate rating'] > 0).sum()
print(f"\n   ⚠️  Ratings = 0.0 (unrated) : {zero_ratings} ({zero_ratings/len(df)*100:.1f}%)")
print(f"   ✅  Ratings > 0.0 (rated)   : {non_zero} ({non_zero/len(df)*100:.1f}%)")
print(f"\n   ⚠️  CLASS IMBALANCE DETECTED: {zero_ratings} restaurants have 0 rating (unrated).")

# ── Visualization ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Level 1 | Task 1: Data Exploration & Preprocessing\nCognifyz Technologies', 
             fontsize=15, fontweight='bold', y=1.01)

colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

# Plot 1: Rating Distribution (Histogram)
ax1 = axes[0, 0]
ax1.hist(df['Aggregate rating'], bins=20, color='#3498db', edgecolor='white', linewidth=0.5)
ax1.set_title('Distribution of Aggregate Ratings', fontweight='bold')
ax1.set_xlabel('Aggregate Rating')
ax1.set_ylabel('Number of Restaurants')
ax1.axvline(df['Aggregate rating'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: {df['Aggregate rating'].mean():.2f}")
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Rating Category Distribution
ax2 = axes[0, 1]
rating_cats = df['Rating text'].value_counts()
bars = ax2.bar(rating_cats.index, rating_cats.values, color=colors[:len(rating_cats)], edgecolor='white')
ax2.set_title('Rating Category Distribution', fontweight='bold')
ax2.set_xlabel('Rating Category')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, rating_cats.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, 
             str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Missing Values
ax3 = axes[1, 0]
all_missing = df.isnull().sum()
all_missing = all_missing[all_missing >= 0]
col_names = [c[:18] for c in all_missing.index]
bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in all_missing.values]
ax3.barh(col_names, all_missing.values, color=bar_colors)
ax3.set_title('Missing Values per Column\n(Red = has missing, Green = clean)', fontweight='bold')
ax3.set_xlabel('Missing Count')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Rated vs Unrated Pie
ax4 = axes[1, 1]
sizes = [zero_ratings, non_zero]
labels = [f'Unrated (0.0)\n{zero_ratings} restaurants', f'Rated (>0)\n{non_zero} restaurants']
explode = (0.05, 0)
ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'],
        explode=explode, startangle=90, textprops={'fontsize': 10})
ax4.set_title('Class Imbalance: Rated vs Unrated', fontweight='bold')

plt.tight_layout()
plt.savefig('level1_task1.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Visualization saved!")
print(f"\n{'='*60}")
print("   TASK 1 COMPLETE ✅")
print(f"{'='*60}")
