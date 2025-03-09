FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv

RUN apt update && \
    apt install -y espeak-ng git git-lfs && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip3 install --upgrade --no-cache-dir torch==2.5.1 torchvision==0.20.1+cu121 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/SparkAudio/Spark-TTS.git /Spark-TTS
WORKDIR /Spark-TTS
RUN pip install -r requirements.txt
RUN mkdir -p pretrained_models
# Make sure you have git-lfs installed (https://git-lfs.com)
RUN git lfs install
RUN git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B

# Install Gradio
RUN pip install gradio

# Make port 7860 available to the world outside this container
EXPOSE 7310

# Copy .env file to the container
COPY .env .env

# Copy src directory to the container
COPY src/ src/

# Set PYTHONPATH
ENV PYTHONPATH=/Spark-TTS

# Run app.py when the container launches
CMD ["sh", "-c", "python -u webui.py --device 0 --server_port ${PORT}"]


