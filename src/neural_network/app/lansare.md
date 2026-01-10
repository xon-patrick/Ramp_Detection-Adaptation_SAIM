# Lansare - Instrucțiuni

- Installare dependențe (recomandat):

```bash
pip install -r requirements.txt
```

```bash
pip install streamlit ultralytics pillow
```

- Lansare aplicație Streamlit:

```bash
python -m streamlit run src/neural_network/app/UI.py
```

- Observații:
  - Asigurați-vă că modelul se află în `src/neural_network/models/best.pt`.
  - Pe Windows, rulați comenzile într-un terminal PowerShell sau CMD cu mediul Python activ.