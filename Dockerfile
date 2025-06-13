FROM alpine:latest AS alpine
# ! We're using alpine as a hack to fetch the musl libraries.

FROM cgr.dev/chainguard/wolfi-base:latest AS base
ENV LANG="C.UTF-8"
ENV UV_FROZEN=true
ENV UV_NO_CACHE=true

WORKDIR /app
RUN chmod 777 /app
RUN apk add --no-cache wget bash tzdata
RUN apk add --no-cache python-3.12-dev
RUN apk add --no-cache uv
COPY --from=alpine --chmod=755 /lib/*musl-*.so.1 /lib/
USER nonroot

FROM base AS final
USER root
RUN apk add --no-cache krb5-dev libpq unixodbc-dev libaio krb5
RUN CURRENT_ARCH=$(uname -m | sed 's|x86_64|amd64|g') \
  && wget https://download.microsoft.com/download/fae28b9a-d880-42fd-9b98-d779f0fdd77f/msodbcsql18_18.5.1.1-1_$CURRENT_ARCH.apk -qO /tmp/msodbcsql.apk \
  && apk add --no-cache --allow-untrusted /tmp/msodbcsql.apk  \
  && rm -rf /tmp/msodbcsql.apk \
  && apk add --no-cache glibc-iconv

RUN CURRENT_ARCH=$(uname -m | sed 's|x86_64|x64|g') \
  && if [ "$CURRENT_ARCH" != "x64" ]; then exit 0; fi \
  && wget https://download.oracle.com/otn_software/linux/instantclient/211000/instantclient-basic-linux.$CURRENT_ARCH-21.1.0.0.0.zip -qO /tmp/oracle-instantclient.zip \
  && wget https://download.oracle.com/otn_software/linux/instantclient/211000/instantclient-sqlplus-linux.$CURRENT_ARCH-21.1.0.0.0.zip -qO /tmp/oracle-sqlplus.zip \
  && unzip /tmp/oracle-instantclient.zip -d /opt/oracle \
  && unzip /tmp/oracle-sqlplus.zip -d /opt/oracle \
  && rm -rf /tmp/oracle-sqlplus.zip /tmp/oracle-instantclient.zip \
  && sh -c "echo '/opt/oracle/instantclient_21_1' > /etc/ld.so.conf.d/oracle-instantclient.conf"
ENV PATH="$PATH:/opt/oracle/instantclient_21_1"
ENV ORACLE_HOME="/opt/oracle/instantclient_21_1"

COPY ./uv.lock ./pyproject.toml ./
RUN apk add --no-cache --virtual temp-build-deps gcc~12 postgresql-dev \
  && uv sync --no-editable --all-extras --no-extra local-gpu \
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
  --no-install-project \
  && apk del temp-build-deps

RUN uv pip install --index-strategy unsafe-first-match torch==2.6.0+cpu torchvision==0.21.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
COPY mostlyai ./mostlyai
COPY README.md ./
RUN uv pip install -e .
COPY ./tools/docker_entrypoint.py ./entrypoint.py

USER nonroot

EXPOSE 8080
ENTRYPOINT [ "uv", "run", "--", "entrypoint.py" ]
