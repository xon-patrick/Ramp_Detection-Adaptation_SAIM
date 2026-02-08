# Dataset – Ramp Detection and Classification (YOLO)

Acest folder contine datele brute colectate cu ROS2, datele procesate si dataset-ul final utilizat pentru antrenarea modelelor YOLO pentru detectarea si clasificarea rampelor.

---

## 1. Continutul folderului `data/`
```bash
data/
├── raw/                 
│   └── iamgini...
├── test/
│   ├── images/                       
│   └── labels/
├── train/
│   ├── images/                     
│   └── labels/
├── valid/
│   ├── images/                     
│   └── labels/
├── data.yaml
├── README.dataset.txt
├── README.roboflow.txt
└── README.md 
```

---

## 2. Informatii despre dataset

### 2.1 Descriere generala

Dataset-ul include imagini capturate de camera robotului in diverse scenarii, cu accent pe detectarea rampelor in fata robotului. Fiecare imagine are asociata o eticheta YOLO cu bounding box si clasa.

Dataset-ul este folosit pentru:
- Detectarea rampelor in timp real pe robotul mobil 4WD
- Clasificarea directiei rampei (urcare/coborare) pentru adaptarea comportamentului
- Identificarea balustradelor de siguranta
- Estimarea pozitiei rampei inainte ca robotul sa ajunga la ea

---

## 2.2 Caracteristicile dataset-ului

- **Numar de clase:** 3
- **Tipuri de date:**
  - Imagini RGB
  - Etichete YOLO (TXT)
- **Format fisiere:**
  - **Imagini:** JPG/PNG (640x480 pixels, RGB)
  - **Labels:** TXT format YOLO (coordonate normalizate)
  - **Rosbag:** DB3 (ROS2 format)
  - **Metadata:** YAML (data.yaml - configurare antrenament) 

---

## 2.3 Clase YOLO

| ID | Nume clasa | Descriere |
|----|------------|-----------|
| 0 | rampDown | Rampa in coborare (descendent) |
| 1 | rampUp | Rampa in urcare (ascendent) |
| 2 | ramps-railing | Balustrada / zona de siguranta rampa |

---

## 3. Procesare date

### 3.1 Extragere imagini din rosbag
Imaginile sunt extrase din topicul camerei:
```bash 
ros2 bag record /camera/compressed_raw
```
### 3.2 Selectarea si curatarea imaginilor
- Stergerea imaginilor fara rampe si a celor nefocusate/blurate
- Pastrarea doar a imaginilor clare din unghiuri relevante pentru navigare
- Eliminarea duplicatelor si a frame-urilor similare

### 3.3 Adnotare YOLO
- Etichetare manuala folosind platforma Roboflow
- Bounding boxes cu coordonate normalizate [x_center, y_center, width, height]
- Export in format YOLOv8 compatibil

---

## 4. Statistici de baza ale dataset-ului

### 4.1 Numar total de imagini
- **Train:** 165 imagini (69.9%)
- **Valid:** 36 imagini (15.3%)
- **Test:** 35 imagini (14.8%)
- **Total:** 236 imagini

### 4.2 Distributia adnotarilor pe clase
- **rampDown (cls 0):** 12 adnotari (1.5%)
- **rampUp (cls 1):** 42 adnotari (5.4%)
- **ramps-railing (cls 2):** 729 adnotari (93.2%)
- **Total:** 783 adnotari

### 4.3 Dezechilibru de clase
⚠️ **Observatie critica:** Dataset-ul prezinta un dezechilibru sever de clase:
- Clasa majoritara `ramps-railing` reprezinta 93.2% din toate adnotarile
- Clasele `rampDown` si `rampUp` (critice pentru navigare) sunt semnificativ subreprezentate
- Acest dezechilibru afecteaza performanta modelului: recall 50% pe `ramps-railing` vs 100% pe `rampDown`

### 4.4 Probleme observate
- Variatii semnificative de iluminare (interior/exterior)
- Unghiuri de capturare limitate
- Dezechilibru sever de clase (93.2% vs 6.8%)

## 5. Performanta modelului pe dataset

### 5.1 Metrici globale (YOLOv8m - optimizat)
- **mAP50:** 80.4%
- **mAP50-95:** 58.7%
- **F1-Score:** 81.3%
- **Precision:** 82.9%
- **Recall:** 80.0%

### 5.2 Performanta pe clase
| Clasa | Precision | Recall | mAP50 | F1-Score |
|-------|-----------|--------|-------|----------|
| rampDown | 95.4% | **100%** | 99.5% | 97.6% |
| rampUp | 95.4% | 89.9% | 91.5% | 92.6% |
| ramps-railing | 58.0% | **50.0%** | 50.3% | 53.7% |

### 5.3 Observatii
✅ **Forte:**
- Detectie perfecta a rampelor critice (rampDown/rampUp: 95%+ recall)
- Precision ridicata pe toate clasele

⚠️ **Limitari:**
- Recall scazut pe `ramps-railing` (50%) din cauza dezechilibrului de clase
- Model prioritizeaza clasele minoritare (esentiale pentru siguranta)

---

## 6. Seturi finale

### Proportii reale:
- **70%** train (165 imagini)
- **15%** val (36 imagini)
- **15%** test (35 imagini)

---

## 7. Versiuni ale dataset-ului

| Versiune | Descriere | Continut |
|----------|-----------|----------|
| v0.1 | Date brute | rosbags .db3 | Arhivat |
| v0.2 | Date brute procesate | imagini .png | Arhivat |
| v0.3 | Date procesate si impartite train/test/val | imagini .png |
| v1 | mai multe ddate adaugate in diferite timpurui ale zilei si cu interferente(persoane ce umbla)| imagini .png|


---

## 8. Recomandari pentru imbunatatire

### 8.1 Prioritate inalta
- **Colectare date suplimentare pentru rampDown/rampUp:** Minim 300 adnotari noi pentru fiecare clasa (de la 12/42 → 300+)
- **Class weighting:** Aplicare ponderi inverse in functia de loss pentru compensarea dezechilibrului

### 8.2 Prioritate medie
- **Augmentare targetata:** Focus pe clasele minoritare (flip, rotation, brightness)
- **Diverse scenarii:** Iluminare variata, unghiuri diferite, conditii meteo

### 8.3 Prioritate scazuta
- **Reducere adnotari ramps-railing:** Subsampling la ~200 adnotari pentru echilibrare
- **Active learning:** Selectie inteligenta a imaginilor pentru adnotare

---

## 9. Notite suplimentare

- Toate imaginile sunt capturate de camera robotului (USB Camera 640x480, RGB)
- Fisierele rosbag (ROS2) permit recrearea dataset-ului
- Adnotare realizata manual folosind platforma Roboflow
- Format export: YOLOv8 (TXT cu coordonate normalizate)
- Dataset utilizat pentru antrenarea modelului deployat: `trained_model_v1.onnx`

