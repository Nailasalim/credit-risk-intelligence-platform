FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
EXPOSE 8000

CMD ["streamlit", "run", "ui/streamlit_app.py"]