FROM python:3.9-slim

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /sbin/nologin -c "Docker image user" appuser
RUN mkdir /home/appuser && chown appuser:appuser /home/appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

RUN mkdir -p /app/app/models && chown -R appuser:appuser /app/app

USER appuser

EXPOSE 8000

# ENV NAME World

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]