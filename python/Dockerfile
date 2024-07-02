#FROM python:3.11.9-bookworm
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN ls -la /app
RUN pip install -r requirements.txt 
#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY . /app
RUN ls -la
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app.py"]