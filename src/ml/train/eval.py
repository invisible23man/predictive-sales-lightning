import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, dataloader, title="Validation", plot=False):
    model.eval()
    preds, targets = [], []

    for x, y in dataloader:
        with torch.no_grad():
            y_hat = model(x)
        preds.extend(y_hat.squeeze().tolist())
        targets.extend(y.squeeze().tolist())

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)

    print(f"✅ {title} - MSE: {mse:.4f} | MAE: {mae:.4f}")

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(targets, label="True")
        plt.plot(preds, label="Predicted")
        plt.title(f"{title} - Actual vs. Predicted")
        plt.xlabel("Time step")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {"mse": mse, "mae": mae}
