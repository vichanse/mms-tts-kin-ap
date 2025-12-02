FROM python:3.10-slim

# Config Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Dossier de travail
WORKDIR /app

# --- 1) Installer torch CPU uniquement (précompilé) ---
# On l'installe AVANT le reste pour profiter du cache Docker
COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.1 && \
    pip install --no-cache-dir -r requirements.txt

# --- 2) Copier le code de l'app ---
COPY . .

# Railway fournit la variable PORT
ENV PORT=8000

# --- 3) Commande de démarrage ---
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
