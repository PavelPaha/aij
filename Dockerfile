FROM cr.ai.cloud.ru/aicloud-base-images/cuda11.8-torch2-py310:0.0.36

USER root

WORKDIR /app

COPY ./checkpoints ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install packaging
RUN pip uninstall -y ninja && pip install ninja
RUN pip install flash-attn --no-build-isolation --no-cache-dir

USER jovyan
WORKDIR /home/jovyan