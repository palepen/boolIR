# Stage 1: Base Image with CUDA 12.1.1 and System Dependencies
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install core system dependencies
# MODIFIED: Removed git and python3-venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ cmake make wget unzip ca-certificates \
    libtinfo-dev zlib1g-dev python3 python3-pip libcudnn9-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# --- Install OpenCilk v2.1.0 from local archive ---
COPY vendor/opencilk.tar.gz /tmp/opencilk.tar.gz

# Instead of running an installer, move the extracted files directly.
RUN cd /tmp && \
    tar -xvf opencilk.tar.gz && \
    mkdir -p /opt/opencilk && \
    mv opencilk-2.1.0-x86_64-linux-gnu-ubuntu-22.04/* /opt/opencilk/ && \
    rm -rf /tmp/*
ENV PATH="/opt/opencilk/bin:${PATH}"

# --- Install ONNX Runtime v1.18.0 from local archive ---
COPY vendor/onnxruntime.tgz /tmp/onnxruntime.tgz

# Install to /opt/onnxruntime to match the Makefile
RUN cd /tmp && \
    tar -zxvf onnxruntime.tgz && \
    mkdir -p /opt/onnxruntime && \
    mv onnxruntime-linux-x64-gpu-1.23.0/* /opt/onnxruntime/ && \
    rm -rf /tmp/*

# --- Environment Setup ---
ENV LD_LIBRARY_PATH="/opt/opencilk/lib:/opt/onnxruntime/lib:${LD_LIBRARY_PATH}"

# --- Install Python Dependencies ---
RUN python3 -m pip install --timeout=600 --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install --timeout=600 --no-cache-dir sentence-transformers
  
RUN python3 -m pip install --timeout=600 --no-cache-dir onnx


# --- Copy and Build Project ---
COPY . .
RUN python3 scripts/export_model.py
RUN make

# Set the default command to a shell
CMD ["bash"]