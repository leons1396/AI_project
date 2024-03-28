import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot(data):
    cols = list(data.columns)
    while True:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            if col == "Label":
                continue
            if i == 2:
                break

            sns.boxplot(data=data, x=col, y="Label", ax=axes[i])
        _ = cols.pop(0)
        _ = cols.pop(0)

        plt.tight_layout()
        plt.show()

        if len(cols) == 0:
            break