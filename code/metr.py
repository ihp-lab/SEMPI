import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

def compute_ccc_batched(y_pred, y_true):
    # Flatten 
    y_true_np = y_true.flatten()
    y_pred_np = y_pred.flatten()
    mean_true = np.mean(y_true_np)
    mean_pred = np.mean(y_pred_np)
    std_true = np.std(y_true_np)
    std_pred = np.std(y_pred_np)

    # Pearson
    rho, _ = pearsonr(y_true_np, y_pred_np)

    # CCC from Pearson
    ccc = (2 * rho * std_true * std_pred) / (std_true**2 + std_pred**2 + (mean_true - mean_pred)**2)
    
    return ccc

def compute_r2_score_batched(y_true, y_pred):
    # Flatten 
    y_true_np = y_true.flatten()
    y_pred_np = y_pred.flatten()

    # Compute R^2 score
    r2 = r2_score(y_true_np, y_pred_np)
    return r2


import numpy as np
from scipy.stats import pearsonr

def compute_pearson_correlation_batched(y_pred, y_true):
    # Flatten 
    y_true_np = y_true.flatten()
    y_pred_np = y_pred.flatten()

    # Calculate Pearson 
    rho, _ = pearsonr(y_true_np, y_pred_np)

    return rho

if __name__ == "__main__":
    # Example usage with batched data   
    y_true_batched = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_pred_batched = torch.tensor([[1.1], [2.2], [2.9], [4.0]])
    ccc_batched = compute_ccc_batched(y_true_batched, y_pred_batched)
    print(f"Batched CCC Score: {ccc_batched}")
    # Example usage with batched data
    y_true_batched = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_pred_batched = torch.tensor([[1.1], [2.2], [2.9], [4.0]])
    r2_batched = compute_r2_score_batched(y_true_batched, y_pred_batched)
    print(f"Batched R2 Score: {r2_batched}")