# Ramp_Detection-Adaptation_SAIM

Propuneri proiect:
A. Identificare si clasificare automata:	Chiar daca robotul functioneaza in medii outdoor sau indoor acesta trebuie sa identifice daca in drumul sau se afla o denivelare ce poate fi parcursa, iar apoi sa o clasifice pentru a putea stii cum sa actioneze: Rampa sau Panta, Dreapta sau Curbata. Aceasta va fi facuta prin estimarea unghiului rampei si compararea cu modelul SIA. Desigur nu toate denivelarile se modifica gradual, astfel clasificarea pooate fi schimbata pe parcurs bazat pe informatii IMU, Camera, LiDar sau encoderele motoarelor.
B. Adaptabilitate
	Bazat pe clasficarea facuta trebuie sa poata adapta corect viteza motoarelor pentru o navigatie cat mai rapida, lina si sigura prin aceste medii. Astfel la coborare sa incetineasca pentru a nu se rasturna, iar la urcare sa mareasca viteza pentru o aderenta mai bune. Desigur aici intra si alti factori, corectaarea vitezei va fi facuta pe baza unghiului de inclinare, tipului de denivelare si daca robotul este sau nu incarcat.
