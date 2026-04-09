FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install fastapi uvicorn openai requests openenv

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
