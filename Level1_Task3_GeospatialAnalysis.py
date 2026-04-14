"""
COGNIFYZ TECHNOLOGIES - DATA SCIENCE INTERNSHIP
Level 1 | Task 3: Geospatial Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Dataset .csv', encoding='utf-8-sig')
df['Cuisines'].fillna('Unknown', inplace=True)

print("=" * 60)
print("   LEVEL 1 | TASK 3: GEOSPATIAL ANALYSIS")
print("=" * 60)

# Filter valid coordinates
geo_df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)].copy()
print(f"\n📍 Restaurants with valid coordinates: {len(geo_df)}")

# ── Distribution across cities/countries ──────────────────────────────────────
print(f"\n🌍 Restaurant Distribution by Country Code (Top 10):")
country_counts = df['Country Code'].value_counts().head(10)
print(country_counts.to_string())

# ── Correlation: Location vs Rating ───────────────────────────────────────────
rated_geo = geo_df[geo_df['Aggregate rating'] > 0]
lat_corr = rated_geo['Latitude'].corr(rated_geo['Aggregate rating'])
lon_corr = rated_geo['Longitude'].corr(rated_geo['Aggregate rating'])

print(f"\n📊 Correlation: Location vs Rating:")
print(f"   Latitude  ↔ Rating : {lat_corr:.4f}")
print(f"   Longitude ↔ Rating : {lon_corr:.4f}")
print(f"\n   📌 Interpretation:")
if abs(lat_corr) < 0.3 and abs(lon_corr) < 0.3:
    print("   Weak correlation — restaurant location alone does not")
    print("   strongly determine its rating. Quality, cuisine, and")
    print("   service matter more than geography.")
else:
    print("   Moderate/Strong correlation detected between location and rating.")

# ── City-wise average rating ───────────────────────────────────────────────────
print(f"\n🏙️ Top 10 Cities by Average Rating:")
city_rating = df[df['Aggregate rating'] > 0].groupby('City')['Aggregate rating'].agg(['mean','count'])
city_rating = city_rating[city_rating['count'] >= 10].sort_values('mean', ascending=False).head(10)
for city, row in city_rating.iterrows():
    print(f"   {city:<25} : {row['mean']:.2f} avg rating ({int(row['count'])} restaurants)")

# ── Visualization ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Level 1 | Task 3: Geospatial Analysis\nCognifyz Technologies',
             fontsize=15, fontweight='bold')

# Plot 1: World Map of Restaurants (scatter)
ax1 = fig.add_subplot(2, 2, (1, 2))
rated = geo_df[geo_df['Aggregate rating'] > 0]
unrated = geo_df[geo_df['Aggregate rating'] == 0]

ax1.scatter(unrated['Longitude'], unrated['Latitude'],
            c='lightgray', s=4, alpha=0.4, label='Unrated')
norm = Normalize(vmin=rated['Aggregate rating'].min(), vmax=rated['Aggregate rating'].max())
sc = ax1.scatter(rated['Longitude'], rated['Latitude'],
                 c=rated['Aggregate rating'], cmap='RdYlGn',
                 norm=norm, s=8, alpha=0.6, label='Rated')
plt.colorbar(sc, ax=ax1, label='Aggregate Rating', shrink=0.8)
ax1.set_title('🗺️ Global Restaurant Locations (Color = Rating)', fontweight='bold', fontsize=13)
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(alpha=0.2)
ax1.set_facecolor('#f0f4f8')

# Plot 2: Top Cities by Restaurant Count
ax2 = fig.add_subplot(2, 2, 3)
top_cities = df['City'].value_counts().head(12)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_cities)))[::-1]
bars = ax2.bar(range(len(top_cities)), top_cities.values, color=colors, edgecolor='white')
ax2.set_xticks(range(len(top_cities)))
ax2.set_xticklabels(top_cities.index, rotation=40, ha='right', fontsize=9)
ax2.set_title('Top Cities by Restaurant Count', fontweight='bold')
ax2.set_ylabel('Number of Restaurants')
for bar, val in zip(bars, top_cities.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', fontsize=8, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Top Cities by Average Rating
ax3 = fig.add_subplot(2, 2, 4)
city_r = df[df['Aggregate rating'] > 0].groupby('City')['Aggregate rating'].agg(['mean','count'])
city_r = city_r[city_r['count'] >= 10].sort_values('mean', ascending=False).head(12)
colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, len(city_r)))[::-1]
bars2 = ax3.bar(range(len(city_r)), city_r['mean'].values, color=colors2, edgecolor='white')
ax3.set_xticks(range(len(city_r)))
ax3.set_xticklabels(city_r.index, rotation=40, ha='right', fontsize=9)
ax3.set_title('Top Cities by Average Rating\n(min 10 restaurants)', fontweight='bold')
ax3.set_ylabel('Average Rating')
ax3.set_ylim(0, 5.2)
for bar, val in zip(bars2, city_r['mean'].values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')
ax3.axhline(y=df[df['Aggregate rating']>0]['Aggregate rating'].mean(),
            color='red', linestyle='--', linewidth=1, label='Overall Avg')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('level1_task3.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Visualization saved!")
print(f"\n{'='*60}")
print("   TASK 3 COMPLETE ✅")
print(f"{'='*60}")
