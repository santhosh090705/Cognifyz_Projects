"""
COGNIFYZ TECHNOLOGIES - DATA SCIENCE INTERNSHIP
Level 1 | Task 2: Descriptive Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Load & Preprocess ──────────────────────────────────────────────────────────
df = pd.read_csv('Dataset .csv', encoding='utf-8-sig')
df['Cuisines'].fillna('Unknown', inplace=True)

print("=" * 60)
print("   LEVEL 1 | TASK 2: DESCRIPTIVE ANALYSIS")
print("=" * 60)

# ── 1. Statistical Measures ────────────────────────────────────────────────────
print("\n📊 Basic Statistical Measures (Numerical Columns):")
num_cols = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
stats = df[num_cols].describe().round(2)
print(stats.to_string())

# ── 2. Categorical Variables Distribution ──────────────────────────────────────
print(f"\n🌍 Country Code Distribution (Top 10):")
country_dist = df['Country Code'].value_counts().head(10)
print(country_dist.to_string())

print(f"\n🏙️ Top 10 Cities by Number of Restaurants:")
city_dist = df['City'].value_counts().head(10)
for city, count in city_dist.items():
    print(f"   {city:<30} : {count}")

# ── 3. Top Cuisines ────────────────────────────────────────────────────────────
print(f"\n🍽️ Top 15 Cuisines by Number of Restaurants:")
cuisine_series = df['Cuisines'].str.split(', ').explode()
top_cuisines = cuisine_series.value_counts().head(15)
for cuisine, count in top_cuisines.items():
    print(f"   {cuisine:<30} : {count}")

# ── 4. Cities with Highest Restaurants ────────────────────────────────────────
print(f"\n🏆 Top 5 Cities with Most Restaurants:")
top_cities = df['City'].value_counts().head(5)
for city, count in top_cities.items():
    print(f"   {city:<25} : {count} restaurants")

# ── Visualization ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Level 1 | Task 2: Descriptive Analysis\nCognifyz Technologies',
             fontsize=15, fontweight='bold')

colors_blue  = ['#1a73e8','#2196F3','#42A5F5','#64B5F6','#90CAF9','#BBDEFB','#1565C0','#0D47A1','#1E88E5','#29B6F6']
colors_green = ['#2e7d32','#388E3C','#43A047','#4CAF50','#66BB6A','#81C784','#A5D6A7','#1B5E20','#2E7D32','#558B2F',
                '#33691E','#827717','#F57F17','#E65100','#BF360C']

# Plot 1: Top 10 Cities
ax1 = axes[0, 0]
city_counts = df['City'].value_counts().head(10)
bars = ax1.barh(city_counts.index[::-1], city_counts.values[::-1], color=colors_blue)
ax1.set_title('Top 10 Cities by Number of Restaurants', fontweight='bold')
ax1.set_xlabel('Number of Restaurants')
for bar, val in zip(bars, city_counts.values[::-1]):
    ax1.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Top 15 Cuisines
ax2 = axes[0, 1]
tc = top_cuisines
bars2 = ax2.barh(tc.index[::-1], tc.values[::-1], color=colors_green)
ax2.set_title('Top 15 Cuisines by Number of Restaurants', fontweight='bold')
ax2.set_xlabel('Number of Restaurants')
for bar, val in zip(bars2, tc.values[::-1]):
    ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Rating Distribution Boxplot
ax3 = axes[1, 0]
num_data = [df[col].dropna() for col in num_cols]
bp = ax3.boxplot(num_data, labels=num_cols, patch_artist=True,
                 boxprops=dict(facecolor='#3498db', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2))
ax3.set_title('Statistical Distribution of Numerical Columns', fontweight='bold')
ax3.set_xticklabels(num_cols, rotation=20, ha='right')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Votes vs Rating scatter
ax4 = axes[1, 1]
rated = df[df['Aggregate rating'] > 0]
sc = ax4.scatter(rated['Votes'], rated['Aggregate rating'],
                 alpha=0.3, c=rated['Aggregate rating'],
                 cmap='RdYlGn', s=10)
plt.colorbar(sc, ax=ax4, label='Rating')
ax4.set_title('Votes vs Aggregate Rating', fontweight='bold')
ax4.set_xlabel('Number of Votes')
ax4.set_ylabel('Aggregate Rating')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('level1_task2.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Visualization saved!")
print(f"\n{'='*60}")
print("   TASK 2 COMPLETE ✅")
print(f"{'='*60}")
