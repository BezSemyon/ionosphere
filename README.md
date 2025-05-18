# TEC Forecasting Model for Japan Region

This project implements a ConvLSTM-based deep learning model for forecasting Total Electron Content (TEC) in the ionosphere over the Japan region. The model is trained on GNSS-derived VISTA TEC maps and is designed to capture spatiotemporal dynamics of ionospheric variations, which are relevant for space weather studies and potential earthquake precursors.

---

## 📍 Region & Data
- **Region:** Japan (cropped TEC grid: 21×24)
- **Data source:** VISTA TEC maps in HDF5 format https://deepblue.lib.umich.edu/data/concern/data_sets/nc580n00z?locale=en
- **Time window:** Sequences of 24 past timesteps (2 hours) are used to predict 1 timestep ahead.

---

## 🧠 Model
- **Architecture:** ConvLSTM Autoencoder with skip connections
- **Loss:** Huber loss
- **Metrics:** MSE, MAE, R²

---

## 🛠 Training
- Earthquake days (Mw ≥ 5.0) and days with severe ionosperic storm caused by solar activity (<=-100 T) are filtered out using `prj.txt`
- Training/validation split: 80% / 20%
- Training uses ReduceLROnPlateau and custom metrics callback

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
   model = load_model("conv_lstm.h5")
