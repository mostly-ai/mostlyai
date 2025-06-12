FROM python:3.11-slim AS base
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=y
WORKDIR /opt/app-root/src/mostlyai/

FROM base AS deps

USER root

RUN groupadd -r nonroot && useradd -r -g nonroot -m nonroot

RUN apt-get update -y \
  && apt-get install -y libaio1 curl gnupg unzip \
  # * PostgreSQL Connector Dependencies
  && apt-get install -y libpq-dev gcc g++ \
  # * Kerberos Dependencies for Hive Connector
  && apt-get install -y libkrb5-dev krb5-user \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN VERSION_ID=$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2) \
  && curl -sSL -o /tmp/packages-microsoft-prod.deb https://packages.microsoft.com/config/debian/$VERSION_ID/packages-microsoft-prod.deb \
  && dpkg -i /tmp/packages-microsoft-prod.deb \
  && apt-get update -y \
  && apt-get install -y unixodbc-dev msodbcsql18 mssql-tools \
  && apt-get clean \
  && rm -f /tmp/packages-microsoft-prod.deb \
  && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/mssql-tools/bin:$PATH"

# * Oracle Connector Dependencies
RUN CURRENT_ARCH=$(uname -m | sed 's|x86_64|x64|g') \
  && if [ "$CURRENT_ARCH" = "x64" ]; then \
  curl https://download.oracle.com/otn_software/linux/instantclient/211000/instantclient-basic-linux.$CURRENT_ARCH-21.1.0.0.0.zip \
  -o /tmp/oracle-instantclient.zip \
  && curl https://download.oracle.com/otn_software/linux/instantclient/211000/instantclient-sqlplus-linux.$CURRENT_ARCH-21.1.0.0.0.zip \
  -o /tmp/oracle-sqlplus.zip \
  && unzip /tmp/oracle-instantclient.zip -d /opt/oracle \
  && unzip /tmp/oracle-sqlplus.zip -d /opt/oracle \
  && sh -c "echo '/opt/oracle/instantclient_21_1' > /etc/ld.so.conf.d/oracle-instantclient.conf" \
  && ldconfig \
  && rm -rf /tmp/* \
  ; fi

ENV PATH="/opt/oracle/opt/oracle/instantclient_21_1:$PATH"
ENV LD_LIBRARY_PATH=/opt/oracle/instantclient_21_1
ENV ORACLE_HOME=/opt/oracle/instantclient_21_1

FROM deps AS build
ENV UV_SYSTEM_PYTHON=1
ENV UV_FROZEN=true
ENV UV_NO_CACHE=true
ENV UV_PROJECT_ENVIRONMENT=/usr/local

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY ./uv.lock ./pyproject.toml ./
RUN uv sync --no-editable --all-extras --no-extra local-gpu \
  --no-install-package torch \
  --no-install-package torchvision \
  --no-install-package torchaudio \
  --no-install-package vllm \
  --no-install-package bitsandbytes \
  --no-install-package nvidia-cudnn-cu12 \
  --no-install-package nvidia-cublas-cu12 \
  --no-install-package nvidia-cusparse-cu12 \
  --no-install-package nvidia-cufft-cu12 \
  --no-install-package nvidia-cuda-cupti-cu12 \
  --no-install-package nvidia-nvjitlink-cu12 \
  --no-install-package nvidia-cuda-nvrtc-cu12 \
  --no-install-package nvidia-curand-cu12 \
  --no-install-package nvidia-cusolver-cu12 \
  --no-install-package nvidia-cusparselt-cu12 \
  --no-install-package nvidia-nccl-cu12 \
  --no-install-package ray \
  --no-install-package cupy-cuda12x \
  --no-install-package triton \
  --no-install-package mostlyai \
  --no-install-project

RUN uv pip install --index-strategy unsafe-first-match torch==2.6.0+cpu torchvision==0.21.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY mostlyai ./mostlyai
COPY README.md ./
RUN uv pip install -e .

COPY ./tools/docker_entrypoint.py /opt/app-root/src/entrypoint.py

USER nonroot
WORKDIR /home/nonroot

EXPOSE 8080
ENTRYPOINT [ "python", "/opt/app-root/src/entrypoint.py" ]
