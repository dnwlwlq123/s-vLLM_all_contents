FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./gateway/
RUN pip install --no-cache-dir -r gateway/requirements.txt

COPY . ./gateway/

CMD ["python", "gateway/main.py"]
