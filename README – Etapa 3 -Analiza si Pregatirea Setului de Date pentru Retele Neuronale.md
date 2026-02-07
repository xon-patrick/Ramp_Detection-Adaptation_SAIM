# Ramp_Detection-Adaptation_SAIM

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Andrei Patrick-Cristian 
**Data:** 25/11/2025  


## Introducere

**Ramp_Detection-Adaptation_SAIM** este un proiect dedicat dezvoltarii unui sistem autonom pentru detectarea si adaptarea la rampe si denivelari in medii variate, folosind simulare si date reale pentru antrenarea retelelor neuronale. Robotul identifica automat tipul denivelarii si adapteaza viteza motoarelor pentru o navigatie sigura si eficienta, atat indoor, cat si outdoor.

Proiectul combina simularea cu **Gazebo**, colectarea de date prin **ROSbag**, si prelucrarea informatiilor provenite de la senzori precum **LiDAR**, **Camera RGB**, **IMU** si **encoderele motoarelor**.


Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   ├── test/              # set de testare
│   ├── README.dataset.txt
│   ├──data.yaml
│   ├── README.roboflow.txt
│   └── README.md
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
└──config/                # fișiere de configurare
 ```
 
---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:**
- Obtinere de informatii folosind senzori reali de pe un Sistem Autonom Mobil Inteligent (Camera RGB, LiDAR, IMU)
- Obtinerea de informatii folosind senzori Simulatii in Gazebo (Camera RGB, LiDAR, IMU)

* **Modul de achiziție:**  Senzori reali & Simulare
* **Perioada / condițiile colectării:** Decembrie 2025 - Ianuarie 2025 (hol public cu diferite iluminari)

### 2.2 Caracteristicile dataset-ului
**Tipuri de date:** Numerice & Imagini
* **Format fișiere:**  .txt / PNG / .db / .mpac 
---

| Caracteristica | Tip | Unitate | Descriere | Domeniu valori |
|----------------|------|---------|-----------|----------------|
| image | imagine | px | Imagine RGB capturata de camera robotului | 640x480 |
| bbox_xcenter | numeric | proportie | Centru bounding box | 0–1 |
| bbox_ycenter | numeric | proportie | Centru bounding box | 0–1 |
| bbox_width | numeric | proportie | Latime box | 0–1 |
| bbox_height | numeric | proportie | Inaltime box | 0–1 |
| class_id | categorial | – | Tip rampa | {0, 1, 2} |

### Clase YOLO:

| ID | Nume clasa | Descriere |
|----|------------|-----------|
| 0 | rampUp | Rampa de urcat |
| 1 | rampDown | Rampa de coborat |
| 2 | railing | Zona inclinata mica / trecere |

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate
- Iluminare variabila intre scene
- Distanta pana la rampa: 0.5–5 metri
- poze de pe rampa

### 3.2 Calitate date
- Blur
- Unele imagini sunt prea populate de persoane
- Necesara eliminarea imaginilor fara rampa vizibila

### 3.3 Probleme identificate
- Reflexii pe suprafete metalice → artefacte YOLO
- Lumina variabila

---

## 4. Preprocesarea Datelor

### 4.1 Curatarea datelor
- Eliminare frame-uri duplicate din rosbag
- Eliminare imagini fara rampa
- Extragerea imaginilor:
```bash
ros2 bag record /camera/compressed_raw /scan
```
### 4.2 Transformarea caracteristicilor
* **Conversie Imagine:** JPG
* **Normalizare:** Min–Max / Standardizare
* **Encoding pentru variabile categoriale**
* **Ajustarea dezechilibrului de clasă**

### 4.3 Structurarea seturilor de date

* 70% – train
* 15% – validation
* 15% – test

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate	

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---
##  6. Stare Etapă 
- [X] Structură repository configurată
- [X] Dataset analizat (EDA realizată)
- [X] Date preprocesate
- [X] Seturi train/val/test generate
- [X] Documentație actualizată în README + `data/README.md`

---

## Obiective principale

### A. Identificare si clasificare automata
- Detectarea denivelarilor pe traseul robotului (rampe, pante, curburi)
- Clasificarea tipului de denivelare pentru a determina strategia de navigatie
- Estimarea unghiului rampei si compararea cu modelul **SIA**
- Adaptarea clasificarii in timp real pe baza datelor provenite de la:
	- **IMU** (accelerometru si giroscop)
	- **Camera RGB**
	- **LiDAR**
	- **Encodere motoare**

### B. Adaptabilitate
- Ajustarea vitezei motoarelor pe baza clasificarii si a conditiilor terenului
- Coborarea rampei: reducerea vitezei pentru stabilitate
- Urcarea rampei: cresterea vitezei pentru aderenta optima
- Factori considerati pentru ajustare:
	- Unghiul de inclinare
	- Tipul de denivelare
	- Sarcina robotului (daca robotul este incarcat)

---

## Colectarea datelor

### 1. Simulare in Gazebo
- Modelul robotului definit in format **URDF**
- Mediile de test variate (rampe, pante, denivelari curbate)
- Simularea fizicii reale si a comportamentului robotului in diferite scenarii

### 2. Inregistrarea datelor cu ROS
- Folosim **ROSbag** pentru a captura datele de la senzori:
	- **LiDAR** → distante si harta mediului
	- **Camera RGB** → imagini pentru detectarea vizuala
	- **Odometrie** (IMU si encodere) → pozitia si orientarea robotului

	## Colectare date:
```bash
rosbag record /camera/rgb/image_raw /lidar/scan /odom /imu/data /wheel/encoders
```

### 3. Structura dataset-ului
Fiecare intrare din dataset include:
- Imagine RGB
- Scanare LiDAR
- Date IMU (acceleratie, giroscop)
- Date encodere (viteza rotilor, pozitia)
- Eticheta denivelare (tip, unghi, clasificare rampa/panta)

---






