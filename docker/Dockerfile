# MCAI training image: CUDA + JDK21 (gym) + Python3.11/torch-cu121 (trainer) + SSH.
# Build from the repo root:  docker build -f docker/Dockerfile -t mcai-train .
#
# The decompiled+patched server sources (minecraft-decomp/server/src) are baked in, so the
# image only COMPILES the gym (./gradlew :server:classes) — it never re-runs the heavy
# `setup` decompile. The Python trainer venv is rebuilt fresh with the pinned torch cu121.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MCAI_HOME=/workspace/mcai
# GraalVM (not a plain JDK) so the gym can be AOT-compiled to a native binary on the box.
ENV JAVA_HOME=/opt/graalvm
ENV PATH="$JAVA_HOME/bin:/usr/bin:${PATH}"

# --- System packages: Python 3.11 (deadsnakes), SSH, build tools, zlib (native-image needs it) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl wget gnupg git unzip \
        build-essential zlib1g-dev openssh-server rsync procps vim less \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# --- GraalVM Community Edition 21 (bundles native-image) -> ./gradlew native works on the box ---
ARG GRAALVM_VERSION=21.0.2
RUN wget -qO /tmp/graalvm.tar.gz \
      "https://github.com/graalvm/graalvm-ce-builds/releases/download/jdk-${GRAALVM_VERSION}/graalvm-community-jdk-${GRAALVM_VERSION}_linux-x64_bin.tar.gz" \
    && tar -xzf /tmp/graalvm.tar.gz -C /opt \
    && ln -s /opt/graalvm-community-openjdk-${GRAALVM_VERSION}* /opt/graalvm \
    && rm /tmp/graalvm.tar.gz \
    && java -version \
    && (native-image --version || echo "WARN: native-image not on PATH — check the GraalVM build")

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
