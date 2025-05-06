from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch


def evaluate_model(
    model, dataloader, plot: bool = False, category: str = "Unknown"
) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            y_hat = model(x)
            preds.extend(y_hat.numpy())
            targets.extend(y.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label="Actual", marker="o")
        plt.plot(preds, label="Predicted", marker="x")
        plt.title(f"Prediction vs Actual for Category: {category}")
        plt.xlabel("Sample")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()

        plot_path = f"checkpoints/eval_plot_{category.lower()}.png"
        plt.savefig(plot_path)
        plt.close()

    return {"mse": mse, "mae": mae}
