FROM python:3.10-slim

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]