import pandas as pd
import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════
df = pd.read_csv('blast_furnace_data.csv')

X = df[['blast_temp', 'coke_rate', 'pci_rate',
        'oxygen_enrichment', 'air_flow',
        'burden_iron_ore', 'moisture']]

y = df[['hot_metal_output', 'fuel_consumption', 'thermal_efficiency']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Total rows   :", len(df))
print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

# ══════════════════════════════════════════
# EVALUATION FUNCTION
# ══════════════════════════════════════════
def evaluate_model(model, model_name):
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_list, rmse_list = [], []
    for i, col in enumerate(y.columns):
        r2   = r2_score(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        r2_list.append(r2)
        rmse_list.append(rmse)
        print(f"  {col}:")
        print(f"    R²   = {r2:.4f}")
        print(f"    RMSE = {rmse:.4f}")
    avg_r2   = np.mean(r2_list)
    avg_rmse = np.mean(rmse_list)
    print(f"  {'─'*40}")
    print(f"  AVERAGE R²   = {avg_r2:.4f}")
    print(f"  AVERAGE RMSE = {avg_rmse:.4f}")
    return model, avg_r2, avg_rmse

# ══════════════════════════════════════════
# DEFINE MODELS
# ══════════════════════════════════════════
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42)),
    "Polynomial Regression": Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())]),
    "XGBoost": MultiOutputRegressor(
        XGBRegressor(n_estimators=100, learning_rate=0.1,
                     max_depth=4, random_state=42, verbosity=0))
}

# ══════════════════════════════════════════
# RUN ALL MODELS
# ══════════════════════════════════════════
results_list = []
trained_models = {}

for name, model in models.items():
    trained, r2, rmse = evaluate_model(model, name)
    results_list.append({"Model": name, "Average R²": r2, "Average RMSE": rmse})
    trained_models[name] = trained

# ══════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════
results_df = pd.DataFrame(results_list).sort_values(
    by="Average R²", ascending=False).reset_index(drop=True)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

best_name  = results_df.iloc[0]['Model']
best_model = trained_models[best_name]
best_r2    = results_df.iloc[0]['Average R²']

print(f"\n🏆 Best Model: {best_name} (Average R² = {best_r2:.4f})")

# ══════════════════════════════════════════
# FEATURE IMPORTANCE — Random Forest
# ══════════════════════════════════════════
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf_model    = trained_models["Random Forest"]
importances = rf_model.feature_importances_

for name_f, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 100)
    print(f"  {name_f:20s} {imp:.4f}  {bar}")

# ══════════════════════════════════════════
# SAVE BEST MODEL + RESULTS
# ══════════════════════════════════════════
lr_model = trained_models["Linear Regression"]
joblib.dump(lr_model, 'best_model.pkl')
results_df.to_csv('model_results.csv', index=False)
print("\nModel saved → best_model.pkl ✅")
print("Results saved → model_results.csv ✅")

# ══════════════════════════════════════════
# VISUALIZATION 1 — Model Comparison Bar Chart
# ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2ecc71' if i == 0 else '#3498db'
          for i in range(len(results_df))]
bars = ax.barh(results_df['Model'], results_df['Average R²'],
               color=colors, edgecolor='white')
ax.set_xlabel('Average R² Score', fontsize=12)
ax.set_title('Model Comparison — Average R² Score', fontsize=14, fontweight='bold')
ax.set_xlim(0, 0.85)
ax.invert_yaxis()
for bar, val in zip(bars, results_df['Average R²']):
    ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved → model_comparison.png ✅")

# ══════════════════════════════════════════
# VISUALIZATION 2 — Actual vs Predicted
# ══════════════════════════════════════════
y_pred_lr = lr_model.predict(X_test)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
targets = ['hot_metal_output', 'fuel_consumption', 'thermal_efficiency']
colors_scatter = ['#e74c3c', '#3498db', '#2ecc71']

for i, (target, color) in enumerate(zip(targets, colors_scatter)):
    actual    = y_test.iloc[:, i].values
    predicted = y_pred_lr[:, i]
    r2        = r2_score(actual, predicted)
    axes[i].scatter(actual, predicted, alpha=0.4, color=color, s=20)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    axes[i].plot([min_val, max_val], [min_val, max_val],
                 'k--', linewidth=1.5, label='Perfect fit')
    axes[i].set_xlabel(f'Actual {target}', fontsize=10)
    axes[i].set_ylabel(f'Predicted {target}', fontsize=10)
    axes[i].set_title(f'{target}\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
    axes[i].legend(fontsize=9)

plt.suptitle('Actual vs Predicted — Linear Regression', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved → actual_vs_predicted.png ✅")

# ══════════════════════════════════════════
# VISUALIZATION 3 — Feature Importance Plot
# ══════════════════════════════════════════
feat_imp = sorted(zip(X.columns, rf_model.feature_importances_),
                  key=lambda x: -x[1])
feat_names, feat_vals = zip(*feat_imp)

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ['#e74c3c', '#e67e22', '#f1c40f',
              '#2ecc71', '#3498db', '#9b59b6', '#95a5a6']
bars = ax.barh(feat_names, feat_vals, color=bar_colors, edgecolor='white')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, feat_vals):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved → feature_importance.png ✅")

# ══════════════════════════════════════════
# OPTIMIZATION
# ══════════════════════════════════════════
print("\n" + "=" * 60)
print("OPTIMIZATION RESULTS")
print("=" * 60)

def optimize_parameters(target='hot_metal_output', maximize=True):
    blast_temps = np.arange(900, 1201, 50)
    coke_rates  = np.arange(300, 601, 50)
    pci_rates   = np.arange(100, 251, 50)
    fixed = {
        'oxygen_enrichment': 5.0,
        'air_flow': 4500,
        'burden_iron_ore': 75,
        'moisture': 2.0
    }
    best_value  = -np.inf if maximize else np.inf
    best_params = {}
    target_idx  = ['hot_metal_output',
                   'fuel_consumption',
                   'thermal_efficiency'].index(target)
    for bt, cr, pr in product(blast_temps, coke_rates, pci_rates):
        row   = pd.DataFrame([{'blast_temp': bt, 'coke_rate': cr,
                                'pci_rate': pr, **fixed}])
        value = lr_model.predict(row)[0][target_idx]
        if (maximize and value > best_value) or \
           (not maximize and value < best_value):
            best_value  = value
            best_params = {'blast_temp': bt, 'coke_rate': cr,
                           'pci_rate': pr, **fixed}
    return best_params, best_value

p1, v1 = optimize_parameters('hot_metal_output', maximize=True)
print(f"\nTo MAXIMIZE Hot Metal Output:")
print(f"  Predicted Output : {v1:.2f} tons/day")
for k, v in p1.items():
    print(f"  {k:22s}: {v}")

p2, v2 = optimize_parameters('fuel_consumption', maximize=False)
print(f"\nTo MINIMIZE Fuel Consumption:")
print(f"  Predicted Fuel   : {v2:.2f} kg/ton")
for k, v in p2.items():
    print(f"  {k:22s}: {v}")

p3, v3 = optimize_parameters('thermal_efficiency', maximize=True)
print(f"\nTo MAXIMIZE Thermal Efficiency:")
print(f"  Predicted Efficiency : {v3:.2f} %")
for k, v in p3.items():
    print(f"  {k:22s}: {v}")