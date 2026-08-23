# GPU-only extension of the common runtime base.  Build scripts/base.Dockerfile
# first, then build this image as base-gpu:latest.
FROM base:latest

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates wget \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-nvrtc-12-3 cuda-cudart-12-3 libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER appuser
WORKDIR /app
