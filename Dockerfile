FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --create-home --uid 1000 app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python train_models.py --out artifacts && chown -R app:app /app
USER app
EXPOSE 8000 8501
# API by default; for the dashboard run:
#   docker run -p 8501:8501 IMAGE streamlit run dashboard.py --server.address=0.0.0.0 --browser.gatherUsageStats=false
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
