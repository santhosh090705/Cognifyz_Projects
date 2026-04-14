"""
COGNIFYZ TECHNOLOGIES - DATA SCIENCE INTERNSHIP
Level 3 | Task 3: Data Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Dataset .csv', encoding='utf-8-sig')
df['Cuisines'].fillna('Unknown', inplace=True)
for col in ['Has Table booking', 'Has Online delivery']:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

rated = df[df['Aggregate rating'] > 0].copy()

print("=" * 60)
print("   LEVEL 3 | TASK 3: DATA VISUALIZATION")
print("=" * 60)

# ── Figure 1: Rating Distributions ────────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
fig1.suptitle('Level 3 | Task 3: Data Visualization — Rating Distributions\nCognifyz Technologies',
              fontsize=14, fontweight='bold')

# 1a. Histogram of all ratings
ax = axes[0, 0]
n, bins, patches = ax.hist(df['Aggregate rating'], bins=25,
                            color='#3498db', edgecolor='white', linewidth=0.6)
for patch, bin_val in zip(patches, bins):
    if bin_val == 0:
        patch.set_facecolor('#e74c3c')
ax.axvline(rated['Aggregate rating'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean (rated): {rated["Aggregate rating"].mean():.2f}')
ax.set_title('Histogram: All Rating Values', fontweight='bold')
ax.set_xlabel('Aggregate Rating')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 1b. Rating text bar chart
ax = axes[0, 1]
rating_counts = df['Rating text'].value_counts()
order = ['Not rated', 'Poor', 'Average', 'Good', 'Very Good', 'Excellent']
order = [o for o in order if o in rating_counts.index]
vals = [rating_counts.get(o, 0) for o in order]
color_map = {'Not rated': '#bdc3c7', 'Poor': '#e74c3c', 'Average': '#e67e22',
             'Good': '#f1c40f', 'Very Good': '#2ecc71', 'Excellent': '#27ae60'}
bar_colors = [color_map.get(o, '#3498db') for o in order]
bars = ax.bar(order, vals, color=bar_colors, edgecolor='white')
ax.set_title('Rating Category Distribution', fontweight='bold')
ax.set_xlabel('Rating Category')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(val), ha='center', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 1c. KDE-like smooth distribution (rated only)
ax = axes[0, 2]
from matplotlib.patches import FancyArrowPatch
x_vals = np.linspace(rated['Aggregate rating'].min(), rated['Aggregate rating'].max(), 300)
hist_vals, bin_edges = np.histogram(rated['Aggregate rating'], bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.fill_between(bin_centers, hist_vals, alpha=0.4, color='#9b59b6')
ax.plot(bin_centers, hist_vals, color='#9b59b6', linewidth=2)
ax.axvline(rated['Aggregate rating'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {rated["Aggregate rating"].mean():.2f}')
ax.axvline(rated['Aggregate rating'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {rated["Aggregate rating"].median():.2f}')
ax.set_title('Rating Density (Rated Restaurants Only)', fontweight='bold')
ax.set_xlabel('Aggregate Rating')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# 1d. Price Range vs Avg Rating
ax = axes[1, 0]
pr_rating = rated.groupby('Price range')['Aggregate rating'].mean()
colors_pr = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
bars = ax.bar(pr_rating.index, pr_rating.values, color=colors_pr[:len(pr_rating)], edgecolor='white')
ax.set_title('Average Rating by Price Range', fontweight='bold')
ax.set_xlabel('Price Range (1=Low → 4=High)')
ax.set_ylabel('Average Rating')
ax.set_ylim(0, 5.2)
ax.set_xticks(pr_rating.index)
ax.set_xticklabels(['Budget\n(1)', 'Moderate\n(2)', 'Premium\n(3)', 'Luxury\n(4)'])
for bar, val in zip(bars, pr_rating.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 1e. Online Delivery vs Rating
ax = axes[1, 1]
delivery_rating = rated.groupby('Has Online delivery')['Aggregate rating'].mean()
labels = ['No Online Delivery', 'Has Online Delivery']
colors_d = ['#e74c3c', '#2ecc71']
bars = ax.bar(labels, delivery_rating.values, color=colors_d, edgecolor='white', width=0.5)
ax.set_title('Average Rating:\nWith vs Without Online Delivery', fontweight='bold')
ax.set_ylabel('Average Rating')
ax.set_ylim(0, 5.2)
for bar, val in zip(bars, delivery_rating.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 1f. Table Booking vs Rating
ax = axes[1, 2]
tb_rating = rated.groupby('Has Table booking')['Aggregate rating'].mean()
labels_tb = ['No Table Booking', 'Has Table Booking']
colors_tb = ['#e67e22', '#3498db']
bars = ax.bar(labels_tb, tb_rating.values, color=colors_tb, edgecolor='white', width=0.5)
ax.set_title('Average Rating:\nWith vs Without Table Booking', fontweight='bold')
ax.set_ylabel('Average Rating')
ax.set_ylim(0, 5.2)
for bar, val in zip(bars, tb_rating.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('level3_task3a.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 1 (Rating Distributions) saved!")

# ── Figure 2: Cuisine & City Comparisons ──────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Level 3 | Task 3: Data Visualization — Cuisine & City Insights\nCognifyz Technologies',
              fontsize=14, fontweight='bold')

# Explode cuisines
cuisine_df = rated.copy()
cuisine_df['Cuisines'] = cuisine_df['Cuisines'].str.split(', ')
cuisine_exploded = cuisine_df.explode('Cuisines')
cuisine_rating = cuisine_exploded.groupby('Cuisines')['Aggregate rating'].agg(['mean','count'])
cuisine_rating = cuisine_rating[cuisine_rating['count'] >= 50]

# 2a. Top 12 cuisines avg rating (horizontal)
ax = axes2[0, 0]
top12 = cuisine_rating.sort_values('mean', ascending=False).head(12)
colors_c = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top12)))[::-1]
bars = ax.barh(top12.index[::-1], top12['mean'].values[::-1], color=colors_c)
ax.set_title('Top 12 Cuisines by Average Rating\n(min 50 restaurants)', fontweight='bold')
ax.set_xlabel('Average Rating')
ax.set_xlim(0, 5.3)
for bar, val in zip(bars, top12['mean'].values[::-1]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
ax.axvline(x=rated['Aggregate rating'].mean(), color='red', linestyle='--', alpha=0.7, label='Overall Avg')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# 2b. Top 12 cities avg rating
ax = axes2[0, 1]
city_rating = rated.groupby('City')['Aggregate rating'].agg(['mean','count'])
city_rating = city_rating[city_rating['count'] >= 10].sort_values('mean', ascending=False).head(12)
colors_ci = plt.cm.Blues(np.linspace(0.4, 0.9, len(city_rating)))[::-1]
bars = ax.barh(city_rating.index[::-1], city_rating['mean'].values[::-1], color=colors_ci)
ax.set_title('Top 12 Cities by Average Rating\n(min 10 restaurants)', fontweight='bold')
ax.set_xlabel('Average Rating')
ax.set_xlim(0, 5.3)
for bar, val in zip(bars, city_rating['mean'].values[::-1]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
ax.axvline(x=rated['Aggregate rating'].mean(), color='red', linestyle='--', alpha=0.7, label='Overall Avg')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# 2c. Votes vs Rating heatmap-style
ax = axes2[1, 0]
ax.scatter(rated['Votes'], rated['Aggregate rating'],
           alpha=0.15, s=8, c=rated['Price range'], cmap='cool')
z = np.polyfit(rated['Votes'], rated['Aggregate rating'], 1)
p = np.poly1d(z)
x_line = np.linspace(rated['Votes'].min(), rated['Votes'].max(), 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend line')
ax.set_title('Votes vs Rating\n(color = Price Range)', fontweight='bold')
ax.set_xlabel('Votes')
ax.set_ylabel('Aggregate Rating')
ax.legend()
ax.grid(alpha=0.3)

# 2d. Correlation heatmap
ax = axes2[1, 1]
num_cols = ['Aggregate rating', 'Average Cost for two', 'Price range', 'Votes',
            'Has Table booking', 'Has Online delivery']
corr_matrix = rated[num_cols].corr()
im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
short_labels = ['Rating', 'Cost/2', 'Price\nRange', 'Votes', 'Table\nBook', 'Online\nDeliv']
ax.set_xticklabels(short_labels, fontsize=9, rotation=30, ha='right')
ax.set_yticklabels(short_labels, fontsize=9)
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha='center', va='center', fontsize=9,
                color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white',
                fontweight='bold')

plt.tight_layout()
plt.savefig('level3_task3b.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 2 (Cuisine & City Insights) saved!")

print(f"\n{'='*60}")
print("   TASK 3 COMPLETE ✅")
print(f"{'='*60}")

print(f"\n{'='*60}")
print("   🎉 ALL TASKS COMPLETE!")
print("   Level 1: Task 1 ✅ | Task 2 ✅ | Task 3 ✅")
print("   Level 3: Task 1 ✅ | Task 2 ✅ | Task 3 ✅")
print(f"{'='*60}")
