# TEC Forecasting Model for Japan Region

This project implements a ConvLSTM-based deep learning model for forecasting Total Electron Content (TEC) in the ionosphere over the Japan region. The model is trained on GNSS-derived VISTA TEC maps and is designed to capture spatiotemporal dynamics of ionospheric variations, which are relevant for space weather studies and potential earthquake precursors.

---

## 📍 Region & Data
- **Region:** Japan (cropped TEC grid: 21×24)
- **Data source:** VISTA TEC maps in HDF5 format
- **Time window:** Sequences of 24 past timesteps (2 hours) are used to predict 1 timestep ahead.

---

## 🧠 Model
- **Architecture:** ConvLSTM Autoencoder with skip connections
- **Input shape:** (24, 21, 24, 1)
- **Output shape:** (1, 21, 24, 1)
- **Loss:** Huber loss
- **Metrics:** MSE, MAE, R²

---

## 🛠 Training
- Earthquake days (Mw ≥ 5.0) are filtered out using `prj.txt`
- Data is normalized to [0, 1] and missing values (`NaN`, `Inf`) are replaced
- Training/validation split: 80% / 20%
- Training is done for 60 epochs using ReduceLROnPlateau and custom metrics callback

---

## 📈 Evaluation
After training, the model logs:
- MSE, MAE, and R² per epoch
- Metric values are saved to `epoch_errors.json`
- Graphs are saved to:  
  - `mse_over_epochs.png`  
  - `mae_over_epochs.png`  
  - `r2_over_epochs.png`

---

## 🔮 Inference
To make a forecast:
1. Prepare a normalized TEC sequence of shape `(1, 24, 21, 24, 1)`
2. Load model with:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model("conv_lstm.h5", compile=False)
