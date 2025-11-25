# Ramp_Detection-Adaptation_SAIM

## Descriere 
**Ramp_Detection-Adaptation_SAIM** este un proiect dedicat dezvoltarii unui sistem autonom pentru detectarea si adaptarea la rampe si denivelari in medii variate, folosind simulare si date reale pentru antrenarea retelelor neuronale. Robotul identifica automat tipul denivelarii si adapteaza viteza motoarelor pentru o navigatie sigura si eficienta, atat indoor, cat si outdoor.

Proiectul combina simularea cu **Gazebo**, colectarea de date prin **ROSbag**, si prelucrarea informatiilor provenite de la senzori precum **LiDAR**, **Camera RGB**, **IMU** si **encoderele motoarelor**.

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






