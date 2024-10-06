# #FROM python:3.11.9-bookworm
# FROM cnstark/pytorch:1.7.1-py3.9.12-cuda11.0.3-ubuntu18.04
# RUN cat /etc/os-release
# RUN pip3 install --upgrade pip
# COPY requirements.txt /app/requirements.txt
# WORKDIR /app
# RUN ls -la /app
# RUN pip3 install -r requirements.txt 
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# COPY . /app
# RUN ls -la
# EXPOSE 5000
# ENTRYPOINT ["python"]
# CMD ["app.py"]

#FROM dustynv/l4t-pytorch:r35.3.1
FROM neichmann/ai-base:latest
# Set environment variables
# Set the working directory
WORKDIR /app
COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

COPY . /app

RUN ls -la
EXPOSE 5000

# Set the entrypoint
ENTRYPOINT [ "python3" ]
CMD ["app.py"]
