# MCAI training image: CUDA + JDK21 (gym) + Python3.11/torch-cu121 (trainer) + SSH.
# Build from the repo root:  docker build -f docker/Dockerfile -t mcai-train .
#
# The decompiled+patched server sources (minecraft-decomp/server/src) are baked in, so the
# image only COMPILES the gym (./gradlew :server:classes) — it never re-runs the heavy
# `setup` decompile. The Python trainer venv is rebuilt fresh with the pinned torch cu121.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MCAI_HOME=/workspace/mcai
ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
ENV PATH="$JAVA_HOME/bin:/usr/bin:${PATH}"

# --- System packages: JDK 21 (Adoptium), Python 3.11 (deadsnakes), SSH, build tools ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl wget gnupg git unzip \
        build-essential openssh-server rsync procps vim less \
    && install -d /etc/apt/keyrings \
    && wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public \
        | gpg --dearmor -o /etc/apt/keyrings/adoptium.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb jammy main" \
        > /etc/apt/sources.list.d/adoptium.list \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        temurin-21-jdk python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${MCAI_HOME}

# --- Python trainer venv: pinned torch cu121 + deps, then the mcai_train package ---
# (copy only what pip needs first, for layer caching)
COPY trainer/pyproject.toml trainer/pyproject.toml
RUN python3.11 -m venv trainer/.venv \
    && trainer/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && trainer/.venv/bin/pip install --no-cache-dir torch==2.5.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && trainer/.venv/bin/pip install --no-cache-dir numpy pyyaml onnx onnxruntime pytest

# --- Copy the repo (lean, per .dockerignore) ---
COPY . ${MCAI_HOME}

# Editable-install the trainer package into the venv.
RUN trainer/.venv/bin/pip install --no-cache-dir -e trainer

# --- Pre-build the gym (compile patched server + resources). Needs network for gradle 8.10
# + the Mojang/maven deps; bakes them into the image so training starts instantly. ---
RUN cd minecraft-decomp \
    && ./gradlew --no-daemon :server:classes \
    && ./gradlew --no-daemon --stop || true

# --- SSH ---
RUN mkdir -p /run/sshd \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/train.sh ${MCAI_HOME}/train.sh
RUN chmod +x /usr/local/bin/entrypoint.sh ${MCAI_HOME}/train.sh

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
