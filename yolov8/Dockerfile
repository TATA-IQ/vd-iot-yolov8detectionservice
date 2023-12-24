FROM python:3.11.4
RUN pip3 install pandas
RUN pip3 install kafka-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
Run pip3 install --upgrade pip
RUN pip3 install opencv-python
RUN pip3 install numpy
RUN chmod +x /bin/sh
Run pip3 install ultralytics
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install requests
RUN pip3 install sqlalchemy
#RUN pip3 install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install imutils
RUN pip3 install PyYAML
RUN pip3 install tqdm
RUN pip3 install seaborn
RUN pip3 install scipy
Run pip3 install glob2
Run pip3 install paramiko
Run pip3 install fastapi==0.99.1
Run pip3 install "uvicorn[standard]"
Run pip3 install protobuf==3.20.*
Run pip3 install ipython
Run pip3 install psutil
Run pip3 install minio
Run pip3 install python-consul
Run pip3 install console-logging
copy app /app
WORKDIR /app
# RUN mkdir /app/logs
cmd ["python","app.py"]