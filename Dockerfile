FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

# System deps: Python 3.12, swig (for Box2D), SDL2 (pygame), build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3-pip \
        swig \
        build-essential \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-mixer-dev \
        libsdl2-ttf-dev \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /app

# Install Python dependencies
# Copy requirements first for better layer caching
COPY requirements.txt .

# Install torch (CPU-only to keep image small; remove --index-url line for GPU)
RUN pip install --no-cache-dir --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements (drake, gymnasium[box2d], pygame, etc.)
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy project source
COPY . .

# Default: run heuristic controller in headless mode
CMD ["python", "lunar_lander.py", "--headless", "--episodes", "5"]
