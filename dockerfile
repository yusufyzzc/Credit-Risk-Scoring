FROM python:3.9-slim

WORKDIR /app

COPY requierement.txt .

RUN pip install --no-cache-dir -r requierement.txt
RUN pip install flask flask-cors scikit-learn numpy joblib

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]