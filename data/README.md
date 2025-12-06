# Dataset – Ramp Detection and Classification (YOLO)

Acest folder contine datele brute colectate cu ROS2, datele procesate si dataset-ul final utilizat pentru antrenarea modelelor YOLO pentru detectarea si clasificarea rampelor.

---

## 1. Continutul folderului `data/`
```bash
data/
├──labels/
│   ├── raw/               
│   ├── processed/         
│   ├── train/             
│   ├── validation/        
│   └── test/
├──labels/
│   ├── raw/               
│   ├── processed/         
│   ├── train/             
│   ├── validation/        
│   └── test/
└──README.md 
```

---

## 2. Informatii despre dataset

### 2.1 Descriere generala

Dataset-ul include imagini capturate de camera robotului in diverse scenarii, cu accent pe detectarea rampelor in fata robotului. Fiecare imagine are asociata o eticheta YOLO cu bounding box si clasa.

Dataset-ul este folosit pentru:
- detectarea rampelor in timp real
- clasificarea tipului de rampa (urcare, coborare, zona inclinata)
- estimarea corecta a rampei inainte ca robotul sa urce pe ea

---

## 2.2 Caracteristicile dataset-ului

- **Numar de clase:** 3
- **Tipuri de date:**
  - Imagini RGB
  - Etichete YOLO (TXT)
- **Format fisiere:**
  - JPG pentru imagini
  - TXT pentru labels YOLO
  - DB3 pentru ROSBAG
  - CSV 

---

## 2.3 Clase YOLO

| ID | Nume clasa | Descriere |
|----|------------|-----------|
| 0 | ramp_up | Rampa in urcare |
| 1 | ramp_down | Rampa in coborare |
| 2 | flat_transition | Zona inclinata mica / trecere |

---

## 3. Procesare date

### 3.1 Extragere imagini din rosbag
Imaginile sunt extrase din topicul camerei:
```bash 
ros2 bag record /camera/compressed_raw
```
### 3.2 Selectarea si curatarea imaginilor potrivite
- stergerea imaginilor fara rampe si a celor nefocusate conform
- pastrarea doar a imaginilor clare din unghiuri diferite

### 3.3 Adnotare Yolo
- Etichetarea manuala folosind Roboflow

---

## 4. Statistici de baza ale dataset-ului

### Distributia claselor (estimare)
- ramp_up: 
- ramp_down: 
- flat_transition:

### Probleme observate
- variatii semnificative de iluminare

## 6. Seturi finale

### Proportii dorite:
- **80%** train  
- **10%** val  
- **10%** test

---

## 7. Versiuni ale dataset-ului

| Versiune | Descriere | Continut |
|----------|-----------|----------|
| v0.1 | Date brute | rosbags |

---

## 8. Notite suplimentare

- Toate imaginile sunt capturate direct de camera robotului (USB Camera 640x480).
- Fisierele rosbag permit recrearea dataset-ului daca se modifica pipeline-ul.

