"""
COGNIFYZ TECHNOLOGIES - DATA SCIENCE INTERNSHIP
Level 3 | Task 1: Predictive Modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Dataset .csv', encoding='utf-8-sig')
df['Cuisines'].fillna('Unknown', inplace=True)

print("=" * 60)
print("   LEVEL 3 | TASK 1: PREDICTIVE MODELING")
print("=" * 60)

# ── Feature Engineering ────────────────────────────────────────────────────────
print("\n🔧 Preparing Features...")

# Binary encode
for col in ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Encode City and Cuisines
le_city = LabelEncoder()
le_cuisine = LabelEncoder()
df['City_enc'] = le_city.fit_transform(df['City'])
df['Cuisine_enc'] = le_cuisine.fit_transform(df['Cuisines'])

# Name/Address length features
df['Name_length'] = df['Restaurant Name'].str.len()
df['Address_length'] = df['Address'].str.len()
df['Cuisine_count'] = df['Cuisines'].str.split(',').apply(lambda x: len(x) if isinstance(x, list) else 1)

# Features & Target (filter out unrated)
rated_df = df[df['Aggregate rating'] > 0].copy()
feature_cols = ['Country Code', 'City_enc', 'Cuisine_enc', 'Average Cost for two',
                'Has Table booking', 'Has Online delivery', 'Price range',
                'Votes', 'Name_length', 'Address_length', 'Cuisine_count']

X = rated_df[feature_cols]
y = rated_df['Aggregate rating']

print(f"   Total rated samples : {len(rated_df)}")
print(f"   Features used       : {len(feature_cols)}")

# ── Train-Test Split ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Training samples    : {len(X_train)}")
print(f"   Testing  samples    : {len(X_test)}")

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
}

results = {}

print(f"\n{'='*60}")
print(f"{'Model':<22} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
print(f"{'─'*60}")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'model': model, 'y_pred': y_pred, 'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"{name:<22} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f}")

print(f"{'─'*60}")
best_model_name = max(results, key=lambda x: results[x]['R2'])
print(f"\n🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

# ── Feature Importance (Random Forest) ────────────────────────────────────────
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n🔑 Top Feature Importances (Random Forest):")
for feat, imp in importances.items():
    bar = '█' * int(imp * 50)
    print(f"   {feat:<25} : {imp:.4f} {bar}")

# ── Visualization ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Level 3 | Task 1: Predictive Modeling\nCognifyz Technologies',
             fontsize=15, fontweight='bold')

colors = ['#3498db', '#e67e22', '#2ecc71']
model_names = list(results.keys())

# Plot 1: Model Comparison - R² Score
ax1 = axes[0, 0]
r2_scores = [results[m]['R2'] for m in model_names]
bars = ax1.bar(model_names, r2_scores, color=colors, edgecolor='white', linewidth=1.5)
ax1.set_title('Model Comparison: R² Score\n(Higher is Better)', fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_ylim(0, 1.1)
for bar, val in zip(bars, r2_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', fontweight='bold')
ax1.tick_params(axis='x', rotation=15)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect = 1.0')
ax1.legend()

# Plot 2: Model Comparison - MAE & RMSE
ax2 = axes[0, 1]
x = np.arange(len(model_names))
width = 0.35
mae_vals  = [results[m]['MAE'] for m in model_names]
rmse_vals = [results[m]['RMSE'] for m in model_names]
bars1 = ax2.bar(x - width/2, mae_vals, width, label='MAE', color='#3498db', edgecolor='white')
bars2 = ax2.bar(x + width/2, rmse_vals, width, label='RMSE', color='#e74c3c', edgecolor='white')
ax2.set_title('Model Comparison: Error Metrics\n(Lower is Better)', fontweight='bold')
ax2.set_ylabel('Error Value')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=15)
for bar, val in zip(bars1, mae_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=9)
for bar, val in zip(bars2, rmse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=9)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Actual vs Predicted (Best Model)
ax3 = axes[1, 0]
best_pred = results[best_model_name]['y_pred']
ax3.scatter(y_test, best_pred, alpha=0.3, s=10, color='#2ecc71')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
ax3.set_title(f'Actual vs Predicted Ratings\n({best_model_name})', fontweight='bold')
ax3.set_xlabel('Actual Rating')
ax3.set_ylabel('Predicted Rating')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Feature Importance
ax4 = axes[1, 1]
top_imp = importances.head(10)
colors4 = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_imp)))[::-1]
bars4 = ax4.barh(top_imp.index[::-1], top_imp.values[::-1], color=colors4)
ax4.set_title('Feature Importance\n(Random Forest)', fontweight='bold')
ax4.set_xlabel('Importance Score')
for bar, val in zip(bars4, top_imp.values[::-1]):
    ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('level3_task1.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Visualization saved!")
print(f"\n{'='*60}")
print("   TASK 1 COMPLETE ✅")
print(f"{'='*60}")
