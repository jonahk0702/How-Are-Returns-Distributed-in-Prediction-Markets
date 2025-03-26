import matplotlib.pyplot as plt
import pandas as pd

def plot_numeric_vs_target(df, target):
    numeric_cols = df.select_dtypes(include='number').columns.drop(target, errors='ignore')
    
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[col], df[target], alpha=0.6)
        plt.xlabel(col)
        plt.ylabel(target)
        plt.title(f'{col} vs {target}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
