# Data acquisition - ROS2 bag processing

## 1. Metoda de generare / achiziție a datelor
Datele utilizate în acest proiect au fost obținute prin intermediul ROS2 bag, folosind funcționalitatea nativă de înregistrare a mesajelor publicate de robot în timpul rulării acestuia într-un scenariu real sau simulat. În timpul execuției, au fost înregistrate mai multe topicuri senzoriale, cu accent principal pe camera RGB, care furnizează informația vizuală necesară pentru detectarea rampei din calea robotului.

Ulterior, datele au fost procesate offline folosind un script Python care accesează direct fișierul `.db3` generat de `ros2 bag`. Scriptul identifică topicul de interes (`/camera/image_raw/compressed`), deserializează mesajele ROS2 de tip imagine comprimată și extrage cadrele la un interval de timp prestabilit. Imaginile sunt decodate folosind `OpenCV` și salvate pe disc sub formă de fișiere `.png`, pentru a fi utilizate ulterior în etapele de antrenare și testare ale rețelei neuronale.

Această abordare permite reproducibilitate, analiză offline și decuplarea procesului de achiziție a datelor de cel de procesare și inferență.

---

## 2. Parametri folosiți în achiziție și procesare

Principalii parametri utilizați în procesul de extragere a datelor sunt:

Fișiere ros2 bag:
`rosbag2_data_ora.db3`
(conține date brute colectate de la senzorii robotului)

Topicul imaginii:
`/camera/image_raw/compressed`
(imagini RGB comprimate, format JPEG/PNG)

Interval de eșantionare cadre (`FRAME_INTERVAL`):
`5.0 secunde`
→ se salvează o imagine la fiecare 5 secunde pentru a reduce redundanța și volumul de date

Format imagine salvat:
`.png` (fără pierderi, potrivit pentru procesare ulterioară)

Director de ieșire:
`data/raw/`

Validări aplicate:

ignorarea mesajelor goale sau corupte

verificarea decodării corecte a imaginii OpenCV

Achiziția inițială a datelor din ros2 bag păstrează frecvența originală a camerei, iar subsampling-ul este realizat ulterior în faza de procesare offline.

---
## 3. Justificarea relevanței datelor pentru problemă

Datele vizuale colectate de la camera RGB sunt esențiale pentru problema abordată, respectiv detecția și urmărirea unei rampe de către un robot mobil folosind o rețea neuronală. Rampa reprezintă un element geometric și vizual distinct, a cărui identificare se bazează pe textură, muchii, perspectivă și variații de lumină, informații care pot fi extrase eficient din imagini RGB.

Utilizarea ros2 bag permite capturarea datelor în condiții realiste de funcționare (variații de iluminare, mișcare, zgomot senzorial), ceea ce crește robustetea setului de date și relevanța acestuia pentru aplicații reale. Extracția cadrelor la intervale regulate reduce corelația temporală excesivă dintre imagini consecutive și produce un set de date mai echilibrat pentru antrenarea și evaluarea modelelor de învățare automată. Astfel, datele obținute sunt direct relevante și adecvate pentru dezvoltarea unui sistem autonom de percepție și navigație în ROS2.
