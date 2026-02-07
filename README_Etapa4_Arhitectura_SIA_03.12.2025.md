# 📘 README – Etapa 4: Arhitectura Completă a Ramp_Detection-Adaptation_SAIM

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Andrei Patrick-Cristian 
**Data:** [Data]   
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este doar definit și compilat (fără antrenare serioasă).**

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Dectarea automata a unei denivelari/rampe pe care un robot trebuie sa o parcurga | Clasificarea daca ramp trebuie urcata sau coborata | RN + UI(sa apara pe harta robotului) |
| Adaptarea motoarelor pentru o coborarea/urcare sigura | Luand date de la IMU va putea adapta motoarele pentru o parcurge sigura a rampei | RN |
---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

#### Tipuri de contribuții acceptate (exemple din inginerie):

Alegeți UNA sau MAI MULTE dintre variantele de mai jos și **demonstrați clar în repository**:

| **Tip contribuție** | **Exemple concrete din inginerie** | **Dovada minimă cerută** |
|---------------------|-------------------------------------|--------------------------|
| **Date generate prin simulare fizică** | • Traiectorii robot in Gazebo<br>• Date colectate de la IMU si camera simulate| |
| **Date achiziționate cu senzori proprii** | • 100 imagini capturate cu cameră montată pe robot<br> 1000 semnale IMU de pe platformă mobilă | Foto setup experimental + frecvență: 5 secunde |
| **Etichetare/adnotare manuală** | • Etichetat manual 80 de imagini cu rampe si balustrada rampei| Fișier Excel cu labels + capturi ecran tool etichetare folosit: Roboflow |


#### Declarație obligatorie în README:
```markdown
### Contribuția originală la setul de date:

**Total observații finale:** [80] (după Etapa 3 + Etapa 4)
**Observații originale:** [80] ([100]%)

**Tipul contribuției:**
[X] Date achiziționate cu senzori proprii  
[X] Etichetare/adnotare manuală  

**Descriere detaliată:**
[Explicați în 2-3 paragrafe cum ați generat datele, ce metode ați folosit, 
de ce sunt relevante pentru problema voastră, cu ce parametri ați rulat simularea/achiziția]
Generarea datelor a fost realizata prin teleoperarea unui robot 4wd ce opereaza pe ros2 Humble. Am inregistrat folosind
```bash 
ros2 bag record 
```
urcarea si coborarea pe rampa de la intrarea in F.I.I.R. si in jurul acesteia. Am verificat filmarile folosind Foxglove si am extras imagini odata la 5 secunde folosind scriptul extract_image.py, le-am ales pe cele mai putin blurate si apoi le-am adnotat manual folosind RoboFlow. Metodele folosite sunte relvante pentru a putea reproduce gasirea si parcurgerea unei rampe de catre robot.

**Locația codului:** `src/data_acquisition/extract_from_db.py` & `src/data_acquisition/extract_from_bag.py`

```
---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

```bash
IDLE → INIT_ROS2 → LOAD_MODEL → WAIT_MISSION →
ACQUIRE_SENSORS (camera / lidar / IMU) →
PREPROCESS_INPUT →
RN_INFERENCE (Ramp Detection) →
  ├─ [No ramp detected] → EXPLORE / NAVIGATE_DEFAULT →
  |                        ACQUIRE_SENSORS (loop)
  |
  └─ [Ramp detected] → VALIDATE_RAMP →
        ├─ [Valid ramp] → ESTIMATE_RAMP_POSE →
        |               PLAN_APPROACH →
        |               FOLLOW_RAMP →
        |               MONITOR_STABILITY →
        |                 ├─ [Ramp lost] → REACQUIRE_RAMP →
        |                 |                ACQUIRE_SENSORS
        |                 |
        |                 ├─ [Ramp completed] → EXIT_RAMP →
        |                 |                    LOG_MISSION →
        |                 |                    WAIT_MISSION
        |                 |
        |                 └─ [Error] → ERROR
        |
        └─ [False positive] → IGNORE_DETECTION →
                              ACQUIRE_SENSORS
       ↓ [Emergency / Sensor failure / Low battery]
     SAFE_STOP → LOG_STATUS → STOP
```

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** | **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | LLB cu VI-uri de generare/achiziție | **MUST:** Produce CSV cu datele voastre (inclusiv cele 40% originale). Cod rulează fără erori și generează minimum 100 samples demonstrative. |
| **2. Neural Network Module** | `src/neural_network/model.py` sau folder dedicat | LLB cu VI-uri RN | **MUST:** Modelul RN definit, compilat, poate fi încărcat. **NOT required:** Model antrenat cu performanță bună (poate avea weights random/inițializați). |
| **3. Web Service / UI** | Streamlit, Gradio, FastAPI, Flask, Dash | WebVI sau Web Publishing Tool | **MUST:** Primește input de la user și afișează un output. **NOT required:** UI frumos, funcționalități avansate. |

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [X] Cod rulează fără erori: `python src/data_acquisition/generate.py` sau echivalent LabVIEW
- [ ] Generează CSV în format compatibil cu preprocesarea din Etapa 3
- [X] Include minimum 40% date originale în dataset-ul final
- [X] Documentație în cod: ce date generează, cu ce parametri

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [X] Arhitectură RN definită și compilată fără erori
- [X] Model poate fi salvat și reîncărcat
- [X] Include justificare pentru arhitectura aleasă (în docstring sau README)
- [ ] **NU trebuie antrenat** cu performanță bună (weights pot fi random)


#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [X] Propunere Interfață ce primește input de la user (formular, file upload, sau API endpoint)
- [X] Includeți un screenshot demonstrativ în `docs/screenshots/`

**Ce NU e necesar în Etapa 4:**
- UI frumos/profesionist cu grafică avansată
- Funcționalități multiple (istorice, comparații, statistici)
- Predicții corecte (modelul e neantrenat, e normal să fie incorect)
- Deployment în cloud sau server de producție

**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.


## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

**Verificare consistență cu Etapa 3:**

```
proiect-rn-Andrei-Patrick/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/  # Din Etapa 3
│   ├── neural_network/
│   └── app/  # UI schelet
├── docs/
│   ├── state_machine.*           #(state_machine.png sau state_machine.pptx sau state_machine.drawio)
│   └── [alte dovezi]
├── models/  # Untrained model
├── config/
├── README.md
├── README_Etapa3.md              # (deja existent)
├── README_Etapa4_Arhitectura_SIA.md              # ← acest fișier completat (în rădăcină)
└── requirements.txt  # Sau .lvproj
```

**Diferențe față de Etapa 3:**
- Adăugat `data/generated/` pentru contribuția dvs originală
- Adăugat `src/data_acquisition/` - MODUL 1
- Adăugat `src/neural_network/` - MODUL 2
- Adăugat `src/app/` - MODUL 3
- Adăugat `models/` pentru model neantrenat
- Adăugat `docs/state_machine.png` - OBLIGATORIU
- Adăugat `docs/screenshots/` pentru demonstrație UI

---

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [X] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [X] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [X] Cod generare/achiziție date funcțional și documentat
- [X] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [X] Diagrama State Machine creată și salvată în `docs/state_machine.*`
- [X] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [X] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [X] Cod rulează fără erori (`python src/data_acquisition/...` sau echivalent LabVIEW)
- [X] Produce minimum 40% date originale din dataset-ul final
- [ ] CSV generat în format compatibil cu preprocesarea din Etapa 3
- [X] Documentație în `src/data_acquisition/README.md` cu:
  - [X] Metodă de generare/achiziție explicată
  - [X] Parametri folosiți (frecvență, durată, zgomot, etc.)
  - [X] Justificare relevanță date pentru problema voastră

### Modul 2: Neural Network
- [X] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [X] README în `src/neural_network/` cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [x] Propunere Interfață ce pornește fără erori (comanda de lansare testată)
- [x] Screenshot demonstrativ în `docs/screenshots/ui_demo.png`
- [X] README în `src/app/` cu instrucțiuni lansare (comenzi exacte)

---

**Predarea se face prin commit pe GitHub cu mesajul:**  
`"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag obligatoriu:**  
`git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`


