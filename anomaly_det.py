import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_absolute_error, roc_curve, auc

# ------------------------------- CONFIG ---------------------------------------
INPUT_STEPS = 24
TARGET_FILE = "2006_01/VISTA_060131.hdf5"
TRAIN_PATTERNS = ["2005_*", "2006_*"]
PRJ_TXT = "prj.txt"
WEIGHTS = "conv_lstm_1405.h5"

# ------------------------------ MODEL -----------------------------------------
def build_model(shape):
    """Builds a ConvLSTM U-Net-like model."""
    inp = layers.Input(shape=shape)

    # Encoder
    x = layers.ConvLSTM2D(128, 3, padding='same', return_sequences=True, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    s128 = x
    x = layers.MaxPooling3D((1, 2, 2))(x)

    x = layers.ConvLSTM2D(64, 3, padding='same', return_sequences=True, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    s64 = x
    x = layers.MaxPooling3D((1, 2, 2))(x)

    x = layers.ConvLSTM2D(32, 3, padding='same', return_sequences=True, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    s32 = x
    x = layers.MaxPooling3D((1, 2, 2))(x)

    x = layers.ConvLSTM2D(16, 3, padding='same', return_sequences=True, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    def align(x, skip):
        dh, dw = skip.shape[2] - x.shape[2], skip.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            x = layers.ZeroPadding3D(((0, 0), (max(dh, 0), 0), (max(dw, 0), 0)))(x)
        elif dh < 0 or dw < 0:
            x = layers.Cropping3D(((0, 0), (-min(dh, 0), 0), (-min(dw, 0), 0)))(x)
        return x

    # Decoder
    x = layers.UpSampling3D((1, 2, 2))(x); x = align(x, s32)
    x = layers.Concatenate()([x, s32])
    x = layers.ConvLSTM2D(32, 3, padding='same', return_sequences=True, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling3D((1, 2, 2))(x); x = align(x, s64)
    x = layers.Concatenate()([x, s64])
    x = layers.ConvLSTM2D(64, 3, padding='same', return_sequences=True, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling3D((1, 2, 2))(x); x = align(x, s128)
    x = layers.Concatenate()([x, s128])
    x = layers.ConvLSTM2D(128, 3, padding='same', return_sequences=False, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    out = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0), loss=Huber())
    return model

# ------------------------------ HELPERS ---------------------------------------
def load_cube(path):
    """Loads and crops TEC data cube from HDF5 file."""
    with h5py.File(path, 'r') as f:
        V = f['data']['VISTA'][:]
    cube = V[45:66, 302:326, :].transpose(2, 0, 1) if V.shape[-1] == 288 else V[:, 45:66, 302:326]
    return cube[132:217][..., np.newaxis]

def is_eq(prj, y, m, d, Mw=5.0):
    """Returns True if the date corresponds to an earthquake above threshold."""
    with open(prj) as f:
        for line in f:
            if not line.strip(): continue
            yy, mm, dd, *rest = [s.strip() for s in line.split(',')]
            if (yy == y or yy == y[-2:]) and mm.zfill(2) == m and dd.zfill(2) == d:
                try: return float(rest[-1]) >= Mw
                except: pass
    return False

def parse_date(fname):
    """Extracts YYYY, MM, DD from filename."""
    s = os.path.basename(fname)[6:-5]
    return "20" + s[:2], s[2:4], s[4:6]

def mae_for(model, cube):
    """Computes MAE between predicted and true frame."""
    x_in = cube[:INPUT_STEPS][np.newaxis, ...]
    x_gt = cube[INPUT_STEPS][np.newaxis, ...]
    pred = model.predict(x_in, verbose=0)
    return mean_absolute_error(x_gt.flatten(), pred.flatten())

def main():
    model = build_model(load_cube(TARGET_FILE)[:INPUT_STEPS].shape)
    model.load_weights(WEIGHTS)

    maes, labels = [], []

    # Evaluate model across dataset
    for pat in TRAIN_PATTERNS:
        for file in sorted(glob.glob(pat + "/VISTA_*.hdf5")):
            cube = load_cube(file)
            if cube.shape[0] < INPUT_STEPS + 1:
                continue
            mae = mae_for(model, cube)
            y, m, d = parse_date(file)
            label = is_eq(PRJ_TXT, y, m, d)
            maes.append(mae)
            labels.append(label)

    maes, labels = np.array(maes), np.array(labels)
    print(f"Total samples: {len(maes)}, EQ: {labels.sum()}, noEQ: {len(maes) - labels.sum()}")

    # ROC AUC Analysis
    fpr, tpr, thr = roc_curve(labels, maes)
    best = np.argmax(tpr - fpr)
    threshold = thr[best]
    auc_score = auc(fpr, tpr)
    print(f"ROC-AUC = {auc_score:.3f}, best threshold = {threshold:.5f}")

    # Plot ROC Curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.scatter(fpr[best], tpr[best], c='red', label=f"Threshold = {threshold:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve (MAE-based Detection)")
    plt.grid(ls='--', alpha=0.4)
    plt.tight_layout()
    plt.legend()
    plt.savefig("roc_curve_plot.png", dpi=200)
    print("Saved: roc_curve_plot.png")

    # Final test on single target
    cube_t = load_cube(TARGET_FILE)
    mae_t = mae_for(model, cube_t)
    alarm = int(mae_t >= threshold)
    gt = is_eq(PRJ_TXT, *parse_date(TARGET_FILE))
    print(f"{TARGET_FILE} — MAE={mae_t:.4f}, DETECT={alarm}, GroundTruth={gt}")

if __name__ == "__main__":
    main()
