# Strategie Antrenare YOLOv8 - Detectie Rampe

## Rezumat

Model YOLOv8m pentru detectia rampelor pe robot 4WD. Dataset: 236 imagini, 3 clase: **rampDown**, **rampUp**, **ramps-railing**.

---

## 1. Analiza Dataset

### Distributie Clase
```
Clasa           Aparitii    Procent
──────────────────────────────────────
ramps-railing   729         69.2%  (majoritate)
rampUp          154         14.6%  (minoritate)
rampDown        119         11.3%  (minoritate)
──────────────────────────────────────
Total           1002        100%   (din 236 imagini)
```

### Concluzii
- **Dezechilibru major**: 69% ramps-railing
- **Dataset mic**: 236 imagini (risc overfitting)
- **Necesita augmentare agresiva**
- **Early stopping obligatoriu**

---

## 2. Alegerea Arhitecturii: YOLOv8m

| Model | Parametri | mAP | Viteza | Memorie |
|-------|-----------|-----|---------|---------|
| YOLOv8n | 3.2M | 50.4% | 2.3ms | 1.5GB |
| **YOLOv8m** | **25.9M** | **61.1%** | **5.9ms** | **2.8GB** |
| YOLOv8l | 43.7M | 64.9% | 10.1ms | 4.2GB |
| YOLOv8x | 68.2M | 66.4% | 14.3ms | 6.1GB |

### De ce YOLOv8m? ✅
- **Nano (n)**: Prea simplu pentru 3 clase
- **Medium (m)**: Balans optim pentru 236 imagini
- **Large/XLarge**: Overfitting garantat pe dataset mic
- **Performanta**: ~170 FPS pe GPU consumer
- **Deployment**: Ruleaza bine pe NVIDIA Jetson si Raspberry Pi 5 (prin ONNX)

---

## 3. Configuratie Antrenare

### Parametri Principali

```python
# Arhitectura
model = 'yolov8m.pt'        # Medium (25.9M parametri)
imgsz = 640                 # Input 640x640

# Training
batch = 16                  # 15 batch-uri/epoca (optim pentru 236 img)
epochs = 100                # Max epoci pentru dataset mic
patience = 20               # Early stopping dupa 20 epoci stagnare

# Learning Rate
lr0 = 0.005                 # Initial (redus fata de default 0.01)
lrf = 0.01                  # Final = 0.00005 (1% din initial)
warmup_epochs = 3           # Crestere lina primele 3 epoci

# Optimizer
optimizer = 'SGD'           # Standard pentru YOLO
momentum = 0.937            # Accelerare convergenta
weight_decay = 0.0005       # Regularizare L2
```

### De ce Batch Size = 16?
- ✅ 15 batch-uri/epoca = gradient estimation bun
- ✅ ~2.8GB memorie GPU (rezonabil)
- ✅ Standard pentru dataset-uri 100-500 imagini
- ❌ Batch=8: prea multa varianta (30 batch/epoca)
- ❌ Batch=32: prea putine update-uri (8 batch/epoca)

### De ce LR = 0.005 (nu 0.01)?
- Dataset mic = sensibilitate mare la learning rate
- Fine-tuning pe greutati pre-antrenate (ImageNet)
- 0.005 previne "catastrophic forgetting"
- Cosine annealing pentru decay lin

### De ce 100 Epoci + Early Stop?
- Dataset < 1000 imagini → 100-150 epoci necesare
- 100 epoci = 2400 treceri prin retea
- Early stop patience=20 → se opreste efectiv la ~75-85 epoci
- Previne overfitting dar permite convergenta completa

---

## 4. Augmentare Agresiva

```python
augmentation = {
    'hsv_h': 0.015,         # ±1.5% schimbare culoare
    'hsv_s': 0.7,           # ±70% saturatie (zi/noapte)
    'hsv_v': 0.4,           # ±40% luminozitate
    'degrees': 20,          # ±20° rotatie (teren accidentat)
    'translate': 0.1,       # ±10% translatare
    'scale': 0.3,           # ±30% zoom (distanta variabila)
    'flipud': 0.5,          # 50% flip vertical
    'fliplr': 0.5,          # 50% flip orizontal
    'mosaic': 1.0,          # YOLO mosaic (combina 4 imagini)
    'perspective': 0.0002,  # Perspective warp
    'shear': 5,             # ±5° shear (inclinare camera)
}
```

### Justificare
- **Robot 4WD**: teren neregulat → camera se inclina ±20°
- **Exterior**: iluminare variabila (soare, umbre, nori)
- **Mosaic**: combina 4 imagini → 4x mai multe obiecte/batch
- **Flip**: simetrie stanga-dreapta obligatorie pentru rampe

---

## 5. Metrici Evaluare

```json
{
  "map50": "Mean Average Precision @ IoU=0.50",
  "map": "Mean Average Precision @ IoU=0.50-0.95",
  "test_f1_macro": "F1 mediat pe toate clasele",
  "test_precision_macro": "Precizie medie",
  "test_recall_macro": "Recall mediu"
}
```

**Nota**: Macro averaging = toate clasele au importanta egala (controleaza bias catre clasa majoritara)

---

## 6. Structura Output

```
project_root/
├── models/
│   └── trained_model_v1.pt          # Model final
├── docs/
│   └── loss_curve.png               # Grafice antrenare
├── results/
│   ├── training_history.csv         # Istoricul epocilor
│   ├── test_metrics.json            # Performanta finala
│   └── config_trained_model_v1.json # Configuratie folosita
└── runs/
    └── trained_model_v1/
        └── weights/
            ├── best.pt              # Cel mai bun checkpoint
            └── last.pt              # Checkpoint final
```

---

## 7. Rulare

```bash
cd src/neural_network/
python train.py
```

**Timp estimat**:
- GPU RTX 3060+: ~15-20 minute
- GPU RTX 3090: ~8-10 minute
- CPU (Nu recomandat): ~40-60 minute

---

## 8. Troubleshooting

### CUDA Out of Memory
```python
# Solutii:
batch = 8               # Reduce batch size
imgsz = 416             # Reduce dimensiune input
```

### Loss nu scade
```python
# Solutii:
lr0 = 0.001             # Reduce learning rate
augmentation['scale'] = 0.5  # Creste augmentare
```

### Nu detecteaza clase minoritare
```python
# Astepta mai mult:
patience = 30           # Creste patience
# SAU:
batch = 8               # Mai multe update-uri/epoca
# SAU colecteaza mai multe date pentru rampDown/rampUp
```

---

## 9. Imbunatatiri Viitoare

1. **Mai multe date**: Target 500+ imagini
2. **Class weights**: Penalizeaza clasa majoritara
3. **Ensemble models**: Combina YOLOv8m + YOLOv8s
4. **Hyperparameter tuning**: Foloseste Optuna pentru cautare automata

---

## 10. Concluzie

Configuratia aleasa balanceaza:
- ✅ Complexitate model (YOLOv8m = optimal pentru 236 img)
- ✅ Gradient estimation stabil (Batch=16)
- ✅ Fine-tuning prudent (LR=0.005)
- ✅ Augmentare agresiva (combat overfitting)
- ✅ Early stopping (opreste la ~75-85 epoci)

**Performanta asteptata**: Accuracy ≥ 75% dupa antrenare completa.

---

**Versiune**: 1.0  
**Data**: 7 Februarie 2026  
**Framework**: YOLOv8 (Ultralytics)  
**Dataset**: Roboflow (3 clase, 236 imagini)
