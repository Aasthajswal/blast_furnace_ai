import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'blast_temp':        np.random.uniform(900, 1200, n),
    'coke_rate':         np.random.uniform(300, 600, n),
    'pci_rate':          np.random.uniform(100, 250, n),
    'oxygen_enrichment': np.random.uniform(2, 6, n),
    'air_flow':          np.random.uniform(3000, 5000, n),
    'burden_iron_ore':   np.random.uniform(60, 80, n),
    'moisture':          np.random.uniform(1, 5, n),
})

df['hot_metal_output'] = (
    2500 +
    0.4  * df['blast_temp'] +
    -0.3 * df['coke_rate'] +
    -0.1 * df['pci_rate'] +
    0.5  * df['oxygen_enrichment'] +
    0.002* df['air_flow'] +
    0.3  * df['burden_iron_ore'] +
    -1.5 * df['moisture'] +
    np.random.normal(0, 20, n)
)

df['fuel_consumption'] = (
    200 +
    0.6  * df['coke_rate'] +
    0.4  * df['pci_rate'] +
    -0.02* df['blast_temp'] +
    -0.5 * df['oxygen_enrichment'] +
    -0.01* df['air_flow'] +
    np.random.normal(0, 15, n)
)

df['thermal_efficiency'] = (
    0.05 * df['blast_temp'] +
    -0.03* df['coke_rate'] +
    0.005* df['air_flow'] +
    0.8  * df['oxygen_enrichment'] +
    -0.4 * df['moisture'] +
    np.random.normal(0, 3, n)
).clip(40, 90)

df.to_csv('blast_furnace_data.csv', index=False)
print("Dataset created! Shape:", df.shape)
print("\nOutput column stats:")
print(df[['hot_metal_output', 'fuel_consumption', 'thermal_efficiency']].describe().round(2))