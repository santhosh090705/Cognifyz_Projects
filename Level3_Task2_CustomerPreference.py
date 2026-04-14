"""
COGNIFYZ TECHNOLOGIES - DATA SCIENCE INTERNSHIP
Level 3 | Task 2: Customer Preference Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Dataset .csv', encoding='utf-8-sig')
df['Cuisines'].fillna('Unknown', inplace=True)

print("=" * 60)
print("   LEVEL 3 | TASK 2: CUSTOMER PREFERENCE ANALYSIS")
print("=" * 60)

# Explode cuisines (each row = one cuisine)
cuisine_df = df.copy()
cuisine_df['Cuisines'] = cuisine_df['Cuisines'].str.split(', ')
cuisine_exploded = cuisine_df.explode('Cuisines')

# ── 1. Cuisine vs Rating ───────────────────────────────────────────────────────
print(f"\n🍽️ Cuisine vs Aggregate Rating (Top 20, min 50 restaurants):")
rated = cuisine_exploded[cuisine_exploded['Aggregate rating'] > 0]
cuisine_rating = rated.groupby('Cuisines')['Aggregate rating'].agg(['mean', 'count'])
cuisine_rating = cuisine_rating[cuisine_rating['count'] >= 50].sort_values('mean', ascending=False)

print(f"\n{'Cuisine':<28} {'Avg Rating':>10} {'Count':>8}")
print("─" * 50)
for cuisine, row in cuisine_rating.head(20).iterrows():
    stars = '★' * int(row['mean'])
    print(f"   {cuisine:<25} {row['mean']:>8.2f}   {int(row['count']):>6}  {stars}")

# ── 2. Most Popular Cuisines by Votes ─────────────────────────────────────────
print(f"\n🗳️ Most Popular Cuisines by Total Votes (Top 15):")
cuisine_votes = cuisine_exploded.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False).head(15)
for cuisine, votes in cuisine_votes.items():
    print(f"   {cuisine:<28} : {votes:,} votes")

# ── 3. Cuisines with Highest Ratings ──────────────────────────────────────────
print(f"\n🏆 Top 10 Cuisines with Highest Average Ratings (min 50):")
top_rated_cuisines = cuisine_rating.head(10)
for cuisine, row in top_rated_cuisines.iterrows():
    print(f"   {cuisine:<28} : {row['mean']:.2f} ⭐  ({int(row['count'])} restaurants)")

# ── Visualization ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle('Level 3 | Task 2: Customer Preference Analysis\nCognifyz Technologies',
             fontsize=15, fontweight='bold')

# Plot 1: Top cuisines by avg rating
ax1 = axes[0, 0]
top_r = cuisine_rating.head(15)
colors1 = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(top_r)))[::-1]
bars1 = ax1.barh(top_r.index[::-1], top_r['mean'].values[::-1], color=colors1)
ax1.set_title('Top 15 Cuisines by Average Rating\n(min 50 restaurants)', fontweight='bold')
ax1.set_xlabel('Average Rating')
ax1.set_xlim(0, 5.5)
for bar, val in zip(bars1, top_r['mean'].values[::-1]):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
ax1.axvline(x=rated['Aggregate rating'].mean(), color='blue', linestyle='--',
            linewidth=1.5, label=f"Overall Avg: {rated['Aggregate rating'].mean():.2f}")
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Top cuisines by total votes
ax2 = axes[0, 1]
top_v = cuisine_votes.head(15)
colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_v)))[::-1]
bars2 = ax2.barh(top_v.index[::-1], top_v.values[::-1], color=colors2)
ax2.set_title('Top 15 Most Popular Cuisines by Total Votes', fontweight='bold')
ax2.set_xlabel('Total Votes')
for bar, val in zip(bars2, top_v.values[::-1]):
    ax2.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=9)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Rating distribution per top cuisine (box plot)
ax3 = axes[1, 0]
top10_cuisines = cuisine_rating.head(10).index.tolist()
box_data = [rated[rated['Cuisines'] == c]['Aggregate rating'].values for c in top10_cuisines]
bp = ax3.boxplot(box_data, patch_artist=True, labels=[c[:15] for c in top10_cuisines])
colors3 = plt.cm.Set3(np.linspace(0, 1, len(top10_cuisines)))
for patch, color in zip(bp['boxes'], colors3):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for median in bp['medians']:
    median.set_color('red')
    median.set_linewidth(2)
ax3.set_title('Rating Distribution: Top 10 Cuisines', fontweight='bold')
ax3.set_ylabel('Aggregate Rating')
ax3.tick_params(axis='x', rotation=40)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Votes vs Rating per cuisine (bubble chart)
ax4 = axes[1, 1]
cuisine_summary = rated.groupby('Cuisines').agg(
    avg_rating=('Aggregate rating', 'mean'),
    total_votes=('Votes', 'sum'),
    count=('Restaurant ID', 'count')
).reset_index()
cuisine_summary = cuisine_summary[cuisine_summary['count'] >= 50].head(30)

scatter = ax4.scatter(
    cuisine_summary['total_votes'],
    cuisine_summary['avg_rating'],
    s=cuisine_summary['count'] * 2,
    c=cuisine_summary['avg_rating'],
    cmap='RdYlGn', alpha=0.7, edgecolors='gray', linewidth=0.5
)
plt.colorbar(scatter, ax=ax4, label='Avg Rating')
# Label top ones
for _, row in cuisine_summary.nlargest(8, 'total_votes').iterrows():
    ax4.annotate(row['Cuisines'][:15], (row['total_votes'], row['avg_rating']),
                fontsize=7, ha='left', va='bottom', color='navy')
ax4.set_title('Cuisine Popularity vs Rating\n(bubble size = restaurant count)', fontweight='bold')
ax4.set_xlabel('Total Votes (Popularity)')
ax4.set_ylabel('Average Rating')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('level3_task2.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Visualization saved!")
print(f"\n{'='*60}")
print("   TASK 2 COMPLETE ✅")
print(f"{'='*60}")
