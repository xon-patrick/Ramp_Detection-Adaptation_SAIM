# Specificatii Masina de Stari - Navigare Rampa

## Prezentare Generala
Acest document descrie masina de stari pentru detectia si navigarea autonoma a rampelor folosind detectie prin retea neuronala ONNX pe un robot ROS2.

## Descrierea Starilor

### 1. IDLE (Navigare Normala)
**Scop:** Starea implicita pentru operatiunea normala a robotului
- Robotul navigheaza normal
- Modelul ONNX proceseaza continuu feed-ul de la camera
- Cauta detectie rampa + balustrada (validare duala)
- Publica obstacole pe harta RViz

**Tranzitie:** → RAMP_DETECTED cand ambele (rampa SI balustrada) sunt detectate

---

### 2. RAMP_DETECTED
**Scop:** Validarea initiala a detectiei rampei
- Declansata cand ambele (rampa SI balustrada) sunt detectate
- Stocheaza datele initiale de detectie
- Se pregateste pentru clasificare

**Tranzitii:**
- → CLASSIFY_RAMP (validare automata)

---

### 3. CLASSIFY_RAMP
**Scop:** Determina tipul rampei si valideaza detectia
- Analizeaza confidenta detectiei
- Determina daca rampa este ascendenta sau descendenta
- Foloseste citirea initiala pitch de la IMU
- Publica locatia rampei pe harta RViz

**Tranzitii:**
- → APPROACH_RAMP_UP daca rampa ascendenta detectata
- → APPROACH_RAMP_DOWN daca rampa descendenta detectata
- → IDLE daca detectie falsa

---

### 4. APPROACH_RAMP_UP
**Scop:** Navigheaza catre intrarea rampei ascendente

**Parametri:**
- Viteza: 30-40% din normal
- Robotul se deplaseaza catre intrarea rampei
- Urmareste continuu pozitia rampei
- Actualizeaza vizualizarea RViz

**Tranzitii:**
- → ALIGN_TO_RAMP cand este aproape de intrarea rampei

---

### 5. ALIGN_TO_RAMP
**Scop:** Faza CRITICA de aliniere pentru intrare sigura

**Parametri:**
- **Viteza: 10-15% (foarte incet)**
- Ajusteaza fin pozitia pentru a centra pe rampa
- Calculeaza unghiul optim de intrare
- Foloseste servoing vizual bazat pe bounding box-ul detectiei

**Tranzitii:**
- → ASCENDING_RAMP cand centrat si unghi OK
- → APPROACH_RAMP_UP daca este necesara realiniere

---

### 6. ASCENDING_RAMP
**Scop:** Urca rampa cu tractiune crescuta

**Parametri:**
- **Viteza: 50-60% din normal**
- **Cuplu: Crestere 30-50%** pentru tractiune
- Urmareste traseul rampei folosind detectii
- Monitorizeaza continuu unghiul pitch de la IMU

**Tranzitii:**
- → RAMP_TURN_UP cand pitch IMU → 0° (sectiune plata detectata)
- → IDLE daca detectia rampei s-a pierdut

---

### 7. RAMP_TURN_UP
**Scop:** Gestioneaza virajul de 90° pe rampa

**Parametri:**
- Detectat cand pitch IMU se apropie de 0° (±5°) in timpul urcarii
- **Viteza: Reducere la 20-30%**
- Executa manevra de viraj 90°

**Tranzitii:**
- → VERIFY_ON_RAMP (automat dupa initierea virajului)

---

### 8. VERIFY_ON_RAMP
**Scop:** Confirma ca robotul este inca pe rampa dupa viraj

**Parametri:**
- **Viteza: 15-20% (foarte incet)**
- Verifica daca rampa este inca detectata dupa viraj

**Tranzitii:**
- → ASCENDING_RAMP daca inca pe rampa
- → EXIT_RAMP daca rampa completata

---

### 9. APPROACH_RAMP_DOWN
**Scop:** Navigheaza catre intrarea rampei descendente

**Parametri:**
- **Viteza: 25-35% din normal**
- Similar cu APPROACH_RAMP_UP dar se pregateste pentru coborare

**Tranzitii:**
- → ALIGN_TO_RAMP_DOWN cand aproape de rampa

---

### 10. ALIGN_TO_RAMP_DOWN
**Scop:** Pozitionare pentru coborare sigura

**Parametri:**
- **Viteza: 10-15% (foarte incet)**
- Centreaza robotul pentru coborare
- Calculeaza unghiul sigur de intrare

**Tranzitii:**
- → DESCENDING_RAMP cand centrat si unghi OK
- → APPROACH_RAMP_DOWN daca este necesara realiniere

---

### 11. DESCENDING_RAMP
**Scop:** Coboara rampa in siguranta

**Parametri:**
- **Viteza: 25-35% din normal**
- **Cuplu: Redus sau foloseste control frana**
- Monitorizeaza pitch IMU (unghi negativ)
- Urmareste cu atentie traseul rampei

**Tranzitii:**
- → EXIT_RAMP cand pitch → 0° (teren plat)
- → IDLE daca detectia rampei s-a pierdut

---

### 12. EXIT_RAMP
**Scop:** Tranzitie inapoi la navigare normala

**Parametri:**
- Sterge parametrii specifici rampei
- Reia viteza normala
- Actualizeaza harta de navigare

**Tranzitii:**
- → IDLE (automat)

---

## Intrari de la Senzori

```
Camera → Model ONNX → {
    ramp_detected: bool,        // rampa detectata
    railing_detected: bool,     // balustrada detectata
    bbox: [x, y, w, h],        // bounding box
    confidence: float           // confidenta detectiei
}

IMU → {
    pitch: float,  // grade - inclinare fata-spate
    roll: float,   // grade - inclinare stanga-dreapta
    yaw: float     // grade - directie
}

Odometry → {
    position: [x, y, z],       // pozitie
    velocity: [vx, vy, vz]     // viteza
}

Senzori Distanta → {
    ramp_distance: float       // distanta pana la rampa
}
```

---

## Parametri de Configurare

```python
# Multiplicatori viteza (relativ la viteza normala)
NORMAL_SPEED = 1.0
APPROACH_SPEED = 0.35      # 35% la apropierea de rampa
ALIGN_SPEED = 0.12         # 12% in timpul alinierii
ASCEND_SPEED = 0.55        # 55% in timpul urcarii
DESCEND_SPEED = 0.30       # 30% in timpul coborarii
TURN_SPEED = 0.25          # 25% in timpul virajelor
VERIFY_SPEED = 0.18        # 18% in timpul verificarii

# Multiplicatori cuplu
NORMAL_TORQUE = 1.0
ASCEND_TORQUE = 1.4        # crestere 40% pentru tractiune
DESCEND_TORQUE = 0.7       # reducere 30% pentru control

# Praguri IMU
FLAT_THRESHOLD = 5.0       # ±5 grade considerat plat
RAMP_PITCH_MIN = 10.0      # pitch minim pentru a considera ca rampa
RAMP_PITCH_MAX = 30.0      # unghi maxim asteptat al rampei

# Praguri detectie
MIN_CONFIDENCE = 0.7       # confidenta minima pentru detectie
RAILING_REQUIRED = True    # necesita detectie balustrada pentru validare
MAX_DETECTION_AGE = 1.0    # maxim secunde de la ultima detectie

# Praguri distanta
APPROACH_DISTANCE = 2.0    # metri - incepe apropierea
ALIGN_DISTANCE = 0.5       # metri - incepe alinierea
```

---

## Consideratii de Siguranta

1. **Cerinta Detectie Duala:** Atat rampa CAT SI balustrada trebuie detectate pentru validare
2. **Reducere Viteza:** Reducere critica a vitezei in faza de aliniere
3. **Monitorizare IMU:** Monitorizare continua a pitch-ului pentru detectarea virajelor si tranzitiilor
4. **Timeout Detectie:** Iesire la IDLE daca detectia s-a pierdut pentru > MAX_DETECTION_AGE secunde
5. **Oprire Urgenta:** Poate tranzitiona din orice stare la IDLE la semnal de urgenta

---

## Vizualizare RViz

Masina de stari publica urmatoarele in RViz:
- Bounding box rampa detectata (marker vizualizare)
- Starea curenta (marker text)
- Traseul planificat pe rampa (marker traseu)
- Orientarea robotului relativ la rampa (marker sageata)

---

## Note de Implementare

- Tranzitiile de stare trebuie loggate pentru debugging
- Toate schimbarile de viteza/cuplu trebuie sa foloseasca ramping lin (nu instant)
- Citirile IMU trebuie filtrate (medie mobila sau filtru Kalman)
- Confidenta detectiei trebuie mediata pe mai multe frame-uri
- Considera adaugarea unei stari STUCK pentru recuperare din erori
