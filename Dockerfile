FROM registry.access.redhat.com/ubi9/python-311:latest AS base

USER root
WORKDIR /workspace
ARG UV_VERSION=0.5.26
ENV UV_NO_CACHE=true

FROM base AS system-deps
# * These random steps in the beginning are necessary to avoid DNF caching issues
# *  on with the gha-k8s approach. Only reproducible in the CI environment.
RUN mkdir -p /var/cache/dnf \
  && dnf clean all \
  && rm -rf /var/cache/dnf/* \
  && dnf update -y \
  # * PostgreSQL Runtime Dependencies
  && dnf install -y postgresql-devel postgresql-libs \
  # * Kerberos Dependencies
  && dnf install -y krb5-workstation krb5-libs \
  && dnf install -y libaio \
  && dnf clean all

RUN mkdir -p /var/cache/dnf \
  && dnf clean all \
  && rm -rf /var/cache/dnf/* \
  && curl https://packages.microsoft.com/config/rhel/9.0/prod.repo > /etc/yum.repos.d/mssql-release.repo \
  && dnf update -y \
  # * ODBC Runtime Dependencies
  && dnf install -y unixODBC-devel \
  # * MSSQL Runtime Dependencies
  && ACCEPT_EULA=y dnf install -y msodbcsql18 \
  && dnf clean all
ENV PATH="/opt/mssql-tools/bin:$PATH"

# * Oracle Database Dependencies
RUN CURRENT_ARCH=$(uname -m | sed 's|x86_64|x64|g') \
  && if [ "$CURRENT_ARCH" = "x64" ]; then \
  curl https://download.oracle.com/otn_software/linux/instantclient/211000/instantclient-basic-linux.$CURRENT_ARCH-21.1.0.0.0.zip \
  -o /tmp/oracle-instanclient.zip \
  && unzip /tmp/oracle-instanclient.zip -d /opt/oracle \
  # * Install Oracle SQLPLUS
  && curl https://download.oracle.com/otn_software/linux/instantclient/211000/instantclient-sqlplus-linux.$CURRENT_ARCH-21.1.0.0.0.zip \
  -o /tmp/oracle-sqlplus.zip \
  && unzip /tmp/oracle-sqlplus.zip -d /opt/oracle \
  # * Setup LD
  && sh -c "echo '/opt/oracle/instantclient_21_1' > /etc/ld.so.conf.d/oracle-instantclient.conf" \
  && ldconfig \
  && rm -rf /tmp/* \
  ; fi

ENV PATH="/opt/oracle/opt/oracle/instantclient_21_1:$PATH"
# Use Oracle thick driver
ENV LD_LIBRARY_PATH=/opt/oracle/instantclient_21_1
ENV ORACLE_HOME=/opt/oracle/instantclient_21_1

FROM system-deps AS python-deps
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="$HOME/.local/bin/:$PATH"

# install mostlyai[local]
ADD ./uv.lock ./pyproject.toml ./mostlyai/
ADD ./docs ./mostlyai/docs/
WORKDIR /workspace/mostlyai
RUN uv sync --frozen --no-editable --all-extras --no-extra local --no-extra local-gpu \
  --no-install-package torch \
  --no-install-package mostlyai
ADD ./mostlyai ./mostlyai/
ADD ./README.md .
RUN uv sync --frozen --all-extras --no-extra local --no-extra local-gpu
RUN uv pip install jupyterlab

# download embedder and store in cache
ENV MOSTLY_HF_HOME=$HOME/.cache
RUN uv run --no-sync python -c \
"import os; from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=os.getenv('MOSTLY_HF_HOME'))"

EXPOSE 8888

WORKDIR /workspace
ADD ./tools/entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
