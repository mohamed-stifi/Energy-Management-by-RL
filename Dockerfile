FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SINERGYM_ENERGY_PLUS_INSTALLATION_COMPLETE=true

# 1. Install system dependencies
# We include python3.12-venv here to optimize layer caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    python3.12 \
    python3-pip \
    python3-dev \
    python3.12-venv \
    libopenblas-dev \
    gfortran \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 3. Install EnergyPlus 24.2.0
# We explicitly set the prefix to /usr/local/EnergyPlus-24-2-0 to avoid naming confusion
RUN wget https://github.com/NREL/EnergyPlus/releases/download/v24.2.0a/EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu24.04-x86_64.sh && \
    chmod +x EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu24.04-x86_64.sh && \
    echo "y" | ./EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu24.04-x86_64.sh --skip-license --prefix=/usr/local/EnergyPlus-24-2-0 && \
    rm EnergyPlus-24.2.0-94a887817b-Linux-Ubuntu24.04-x86_64.sh

# Update PATH to match the specific folder defined above
ENV PATH="/usr/local/EnergyPlus-24-2-0:${PATH}"

# 4. Setup Workspace
WORKDIR /workspaces/energy-rl-project

# Copy requirements
COPY requirements.txt .

# 5. Create Virtual Environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 6. Install Python Dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir git+https://github.com/AlejandroCN7/opyplus.git@master && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 6006 8888

# 7. FIX: Copy pyenergyplus to venv
# The path now matches exactly what we defined in step 3
RUN cp -r /usr/local/EnergyPlus-24-2-0/pyenergyplus /opt/venv/lib/python3.12/site-packages/

CMD ["/bin/bash"]