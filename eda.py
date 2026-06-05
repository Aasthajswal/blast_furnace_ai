import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data load karo
df = pd.read_csv('blast_furnace_data.csv')

# Basic info
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nStatistics:")
print(df.describe())

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='black')
    axes[i].set_title(col, fontsize=11, pad=12)
    axes[i].tick_params(axis='x', labelsize=8)
    axes[i].tick_params(axis='y', labelsize=8)

axes[10].set_visible(False)
axes[11].set_visible(False)

plt.suptitle('Blast Furnace - All Features Distribution',
             fontsize=15, y=1.03)
plt.subplots_adjust(top=0.93, hspace=0.6, wspace=0.4)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()