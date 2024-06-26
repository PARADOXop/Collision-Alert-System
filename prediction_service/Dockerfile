FROM python:3.8-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --upgrade tensorflow streamlit
RUN pip install --upgrade pip
CMD ["streamlit", "run", "app.py"]
