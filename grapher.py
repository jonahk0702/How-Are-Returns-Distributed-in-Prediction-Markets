import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_numeric_vs_target_block_means(df, target):
    """
    For each numeric column in df (other than target), make a scatter plot vs. target,
    add a horizontal line at the target mean, then divide the x-axis into 10 bins and
    plot the mean (x, y) points within each bin, connecting them with a line.
    """
    
    # Print overall stats about target for reference
    print("Target mean:", df[target].mean())
    print("Target describe:\n", df[target].describe())
    
    # Identify numeric columns (excluding target)
    numeric_cols = df.select_dtypes(include='number').columns.drop(target, errors='ignore')

    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        
        # Scatter plot of x=col vs y=target
        plt.scatter(df[col], df[target], alpha=0.6, label=col)
        
        # Horizontal line at overall target mean
        mean_val = df[target].mean()
        plt.axhline(y=mean_val, color='gray', linestyle='--', label='Overall Target Mean')
        
        # Divide col into 10 bins. pd.cut(...) creates a categorical bin index (0 to 9)
        df_temp = df[[col, target]].copy()
        df_temp['bin_idx'] = pd.cut(df_temp[col], bins=10, labels=False, include_lowest=True)
        
        # Compute mean x and mean y within each bin
        # (as_index=False ensures we get a DataFrame, not a Series)
        bin_means = df_temp.groupby('bin_idx', as_index=False)[[col, target]].mean()
        
        # We might drop rows where the bin was empty (no data in that interval)
        bin_means.dropna(inplace=True)
        
        # Plot the mean points and connect them with a line
        plt.plot(bin_means[col], bin_means[target], 'o-', color='red', label='Block Means')
        
        # Final plot cosmetics
        plt.xlabel(col)
        plt.ylabel(target)
        plt.title(f"{col} vs. {target} (Block Means)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
