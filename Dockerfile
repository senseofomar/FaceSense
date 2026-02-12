From python:3.10

workdir /app

copy requirements.txt .

run pip install --no-cache-dir -r requirements.txt 

copy . .

expose 8501

cmd ["streamlit","run","app.py","--server.address=0.0.0.0"]
