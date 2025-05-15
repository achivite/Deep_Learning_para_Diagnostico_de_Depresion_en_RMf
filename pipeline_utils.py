import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2
from tqdm import tqdm
from nilearn.masking import compute_epi_mask
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def load_and_crop_volumes(file_path, discard_volumes=5):
    img = nib.load(file_path)
    data = img.get_fdata()
    if data.shape[3] <= discard_volumes:
        return None
    return data[..., discard_volumes:]

def compute_brain_mask(mean_img_3d):
    mean_nifti = nib.Nifti1Image(mean_img_3d, affine=np.eye(4))
    mask_img = compute_epi_mask(mean_nifti)
    return mask_img.get_fdata().astype(bool)

def reconstruct_volume(masked_data, mask):
    T = masked_data.shape[0]
    vol_shape = mask.shape + (T,)
    volume_4d = np.zeros(vol_shape, dtype=np.float32)
    volume_4d[mask] = masked_data.T
    return volume_4d

def select_high_variance_slice(volume_4d):
    variances = [np.var(volume_4d[:, :, z, :]) for z in range(volume_4d.shape[2])]
    return int(np.argmax(variances))

def normalize_zscore_2d(slice_2d):
    mean = np.mean(slice_2d)
    std = np.std(slice_2d)
    return (slice_2d - mean) / std if std != 0 else slice_2d - mean
    
def create_dataset(X, y, batch_size=4, shuffle=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
def plot_training(history):
    metric_keys = ['loss', 'accuracy', 'recall', 'precision', 'auc']
    metric_names = ['Pérdida', 'Exactitud', 'Sensibilidad', 'Precisión', 'AUC']

    plt.figure(figsize=(20, 4))

    for i, (key, name) in enumerate(zip(metric_keys, metric_names)):
        plt.subplot(1, 5, i + 1)
        plt.plot(history.history[key], label='Train')
        plt.plot(history.history[f'val_{key}'], label='Val')
        plt.title(name)
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def model_metrics(modelo, X_val, y_val, umbral=0.5):
    y_prob = modelo.predict(X_val).flatten()
    y_pred = (y_prob >= umbral).astype(int)

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    exactitud = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_val, y_pred)
    sensibilidad = recall_score(y_val, y_pred)
    especificidad = tn / (tn + fp)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print("\nMétricas de evaluación:")
    print(f"{'Exactitud':}: {exactitud:.4f}")
    print(f"{'Precisión':}: {precision:.4f}")
    print(f"{'Sensibilidad':}: {sensibilidad:.4f}")
    print(f"{'Especificidad':}: {especificidad:.4f}")
    print(f"{'Puntuación F1':}: {f1:.4f}")
    print(f"{'Área bajo la curva':}: {auc:.4f}\n")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=["Negativo", "Positivo"],
                yticklabels=["Negativo", "Positivo"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.show()
    

def model_metrics_dataset(modelo, dataset, umbral=0.5):
    y_prob = modelo.predict(dataset).flatten()
    y_pred = (y_prob >= umbral).astype(int)

    y_true = np.concatenate([y for _, y in dataset], axis=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    exactitud = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_true, y_pred)
    sensibilidad = recall_score(y_true, y_pred)
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print("\nMétricas de evaluación:")
    print(f"Exactitud:     {exactitud:.4f}")
    print(f"Precisión:     {precision:.4f}")
    print(f"Sensibilidad:  {sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")
    print(f"F1-score:      {f1:.4f}")
    print(f"AUC:           {auc:.4f}")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=["Negativo", "Positivo"],
                yticklabels=["Negativo", "Positivo"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.show()