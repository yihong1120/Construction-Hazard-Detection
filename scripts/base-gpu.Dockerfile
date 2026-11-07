# Build the common Python 3.15 base first, then this GPU extension.
ARG BASE_IMAGE=base:latest
FROM ${BASE_IMAGE} AS cuda-runtime

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates wget \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-nvrtc-13-0 cuda-cudart-13-0 libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM cuda-runtime AS native-build
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git cuda-cudart-dev-13-0 cuda-crt-13-0 cuda-cccl-13-0 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda-13.0/targets/x86_64-linux
COPY deploy/python315 /app/deploy/python315
ARG NATIVE_BUILD_JOBS=4
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/tmp/native-build \
    python deploy/python315/bootstrap.py --build-only --jobs ${NATIVE_BUILD_JOBS} --cache-dir /tmp/native-build \
    && mkdir -p /opt/native-libs \
    && cp -a /tmp/native-build/build-env/lib/python3.15/site-packages/tensorrt_libs \
        /tmp/native-build/build-env/lib/python3.15/site-packages/tensorrt_cu13_libs-11.4.0.106.dist-info \
        /opt/native-libs/

FROM cuda-runtime
# Keep the official TensorRT libraries already downloaded for compilation.
# Copy their wheel metadata as well so the installer can verify dependencies.
COPY --from=native-build /opt/native-libs /usr/local/lib/python3.15/site-packages/
COPY --from=native-build /app/wheels/python315 /opt/python315-wheels
RUN uv pip install --system --find-links /opt/python315-wheels \
        --index https://pypi.nvidia.com tensorrt==11.4.0.106 onnxruntime==1.31.0 \
    && uv pip check --system \
    && rm -rf /root/.cache/pip /root/.cache/uv /opt/python315-wheels

USER appuser
WORKDIR /app
