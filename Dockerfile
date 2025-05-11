FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m spacy download fr_core_news_sm
COPY . .
RUN python train_model.py 
CMD ["python", "app.py"]