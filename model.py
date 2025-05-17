import os, h5py, json, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============ tensorflow acceleration ============
tf.config.optimizer.set_jit(True)
for g in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

# ---------- helpers ----------
def parse_vista_date(fname):
    s = os.path.basename(fname)[6:-5]
    return "20"+s[:2], s[2:4], s[4:6]

def check_eq(prj,y,m,d, Mw=5.0):
    with open(prj) as f:
        for ln in f:
            if not ln.strip(): continue
            yy,mm,dd,*rest = [p.strip() for p in ln.split(',')]
            if (yy==y or yy==y[-2:]) and mm.zfill(2)==m and dd.zfill(2)==d:
                try: return float(rest[-1])>=Mw
                except: pass
    return False

def load_vista(path):
    with h5py.File(path,'r') as f:
        V = f['data']['VISTA'][:]
    cube = V[45:66,302:326,:].transpose(2,0,1) if V.shape[-1]==288 \
        else V[:,45:66,302:326]
    return cube[132:217][...,np.newaxis]          

def seq(arr, n_in=24, n_out=1):
    X,Y=[],[]
    for i in range(len(arr)-n_in-n_out+1):
        X.append(arr[i:i+n_in]); Y.append(arr[i+n_in:i+n_in+n_out])
    return np.array(X), np.array(Y)

# ---------- ConvLSTM 128‑64‑32‑16‑32‑64‑128 ----------
def build_model(shape):
    inp = layers.Input(shape=shape)        

    x  = layers.ConvLSTM2D(128,3,padding='same',return_sequences=True,
                           activation='relu')(inp)
    x  = layers.BatchNormalization()(x)
    s128 = x                                  # skip‑1
    x  = layers.MaxPooling3D((1,2,2))(x)      # H/2,W/2

    x  = layers.ConvLSTM2D(64,3,padding='same',return_sequences=True,
                           activation='relu')(x)
    x  = layers.BatchNormalization()(x)
    s64 = x                                   # skip‑2
    x  = layers.MaxPooling3D((1,2,2))(x)      # H/4,W/4

    x  = layers.ConvLSTM2D(32,3,padding='same',return_sequences=True,
                           activation='relu')(x)
    x  = layers.BatchNormalization()(x)
    s32 = x                                   # skip‑3
    x  = layers.MaxPooling3D((1,2,2))(x)      # H/8,W/8

    x  = layers.ConvLSTM2D(16,3,padding='same',return_sequences=True,
                           activation='relu')(x)
    x  = layers.BatchNormalization()(x)

    # ---------- helper: align sizes ----------
    def align(x, skip):
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:      
            x = layers.ZeroPadding3D(((0,0),(max(dh,0),0),
                                       (max(dw,0),0)))(x)
        elif dh < 0 or dw < 0:        
            x = layers.Cropping3D(((0,0),(-min(dh,0),0),
                                    (-min(dw,0),0)))(x)
        return x

    x  = layers.UpSampling3D((1,2,2))(x)
    x  = align(x, s32)
    x  = layers.Concatenate()([x, s32])
    x  = layers.ConvLSTM2D(32,3,padding='same',return_sequences=True,
                           activation='relu')(x)
    x  = layers.BatchNormalization()(x)

    x  = layers.UpSampling3D((1,2,2))(x)
    x  = align(x, s64)
    x  = layers.Concatenate()([x, s64])
    x  = layers.ConvLSTM2D(64,3,padding='same',return_sequences=True,
                           activation='relu')(x)
    x  = layers.BatchNormalization()(x)

    x  = layers.UpSampling3D((1,2,2))(x)
    x  = align(x, s128)
    x  = layers.Concatenate()([x, s128])
    x  = layers.ConvLSTM2D(128,3,padding='same',return_sequences=False,
                           activation='relu')(x)
    x  = layers.BatchNormalization()(x)

    out = layers.Conv2D(1,3,padding='same',activation='sigmoid')(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0),
                  loss=Huber())
    model.summary()
    return model



# ---------- callback for metrics ----------
class MetricsCB(tf.keras.callbacks.Callback):
    def __init__(self, Xv,Yv):
        super().__init__(); self.Xv,self.Yv=Xv,Yv; self.log=[]
    def on_epoch_end(self, epoch, logs=None):
        pr = self.model.predict(self.Xv, verbose=0)
        y = self.Yv[:,0]; pr = pr
        mse = mean_squared_error(y.flatten(),pr.flatten())
        mae = mean_absolute_error(y.flatten(),pr.flatten())
        try: r2=r2_score(y.flatten(),pr.flatten())
        except: r2=np.nan
        self.log.append(dict(epoch=epoch+1,mse=mse,mae=mae,r2=r2))
        print(f"[{epoch+1}] MSE={mse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    def on_train_end(self,_):
        json.dump(self.log, open("epoch_errors.json","w"), indent=2)

def main():

    base   = os.path.dirname(os.path.abspath(__file__))
    prj    = os.path.join(base,"prj.txt")

    raws=[]
    for yr in [2005,2006]:
        for m in range(1,13):
            fold=f"{yr}_{m:02d}"
            if not os.path.isdir(fold): continue
            for fn in sorted(os.listdir(fold)):
                if not fn.startswith("VISTA_"): continue
                y,mo,d=parse_vista_date(fn)
                if check_eq(prj,y,mo,d): continue
                try: raws.append(load_vista(os.path.join(fold,fn)))
                except: pass

    data=np.concatenate(raws).astype('float32')
    data=np.nan_to_num(data)
    data=(data-data.min())/(data.max()-data.min()+1e-8)

    X,Y=seq(data,24,1); idx=int(0.8*len(X))
    Xtr,Xv,Ytr,Yv = X[:idx],X[idx:],Y[:idx],Y[idx:]

    model=build_model(X.shape[1:])
    ds_tr=tf.data.Dataset.from_tensor_slices((Xtr,Ytr[:,0])).shuffle(len(Xtr)).batch(256)
    ds_v =tf.data.Dataset.from_tensor_slices((Xv ,Yv[:,0])).batch(256)

    cbs=[
        tf.keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=.5,min_lr=1e-6,verbose=1),
        MetricsCB(Xv,Yv)
    ]
    model.fit(ds_tr,epochs=60,validation_data=ds_v,callbacks=cbs)

    # ===== graphics =====
    log=json.load(open("epoch_errors.json"))
    epochs=[e['epoch'] for e in log]
    for m in ['mse','mae','r2']:
        plt.figure(); plt.plot(epochs,[e[m] for e in log],marker='o')
        plt.title(f"{m.upper()} over epochs"); plt.xlabel("epoch"); plt.ylabel(m.upper())
        plt.grid(ls='--',alpha=.4); plt.tight_layout()
        plt.savefig(f"{m}_over_epochs.png",dpi=200); plt.close()

    model.save("conv_lstm.h5")

if __name__=="__main__":
    main()
