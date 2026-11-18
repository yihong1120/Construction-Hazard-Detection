# Compile the small native wheel independently of the CUDA runtime.
FROM rust:1.93.1-slim-bookworm AS postprocess-toolchain
FROM python:3.15-slim-bookworm AS rust-postprocess-wheel
COPY --from=postprocess-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=postprocess-toolchain /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY native/postprocess /build/native/postprocess
RUN pip install --no-cache-dir maturin==1.16.0 \
    && maturin build --release --locked \
        --manifest-path /build/native/postprocess/Cargo.toml --out /wheels

# Build against the same pinned TensorRT SDK as base-gpu, without a GPU.
FROM python:3.15-slim-bookworm AS rust-inference-wheel
COPY --from=postprocess-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=postprocess-toolchain /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda-13.0/targets/x86_64-linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates wget git g++ \
        libopencv-core-dev libopencv-imgproc-dev \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-dev-13-0 cuda-crt-13-0 cuda-cccl-13-0 \
        cuda-nvrtc-dev-13-0 cuda-driver-dev-13-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.9.9
WORKDIR /build
ARG NATIVE_GPU_MEDIA=0
COPY native/inference /build/native/inference
COPY deploy/python315/bootstrap.py /build/deploy/python315/bootstrap.py
COPY scripts/build_native_inference.py /build/scripts/build_native_inference.py
RUN if [ "$NATIVE_GPU_MEDIA" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libavcodec-dev libavutil-dev \
        && python -m scripts.build_native_inference --output /wheels --gpu-media; \
    else python -m scripts.build_native_inference --output /wheels; fi

# The stream processor uses TensorRT/YOLO, so only this image inherits CUDA.
FROM base-gpu:latest

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.15/site-packages/tensorrt_libs:/usr/local/lib/python3.15/site-packages/nvidia/cu13/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# The shared base image ends as appuser; package installation needs root.
USER root

# Set the working directory in the container
WORKDIR /app

COPY --from=rust-postprocess-wheel /wheels /tmp/postprocess-wheels
COPY --from=rust-inference-wheel /wheels /tmp/inference-wheels
RUN uv pip install --system /tmp/postprocess-wheels/*.whl \
    && uv pip install --system /tmp/inference-wheels/*.whl \
    && python3 -c "import construction_postprocess as n; assert n.ABI_VERSION == 1" \
    && rm -rf /tmp/postprocess-wheels /tmp/inference-wheels

RUN uv export \
    --quiet \
    --format=requirements-txt \
    --no-dev \
    --no-emit-project \
    --frozen \
    --extra streaming \
    --extra yolo \
    --extra yolo-gpu \
    -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip /root/.cache/uv /tmp/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg fonts-noto-core fonts-noto-cjk libglib2.0-0 libgl1 \
        libopencv-core406 libopencv-imgproc406 \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files into the container
COPY --chown=appuser:appuser config /app/config
COPY --chown=appuser:appuser examples /app/examples
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser main.py /app/main.py

# Ensure 'appuser' exists (if not already present in base image)
RUN groupadd -g 1001 appuser && useradd -u 1001 -g appuser -m appuser || true

# Create the 'logs' directory and set ownership
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs
RUN install -d -m 0700 -o appuser -g appuser /app/outputs/event-outbox
RUN install -d -m 0700 -o appuser -g appuser /app/outputs/gpu-resources

# Switch to the non-root user
USER appuser

# Set ENTRYPOINT to allow dynamic arguments for the configuration file
ENTRYPOINT ["python3", "main.py"]

# Default CMD provides a placeholder configuration file
CMD ["--config", "/app/config/configuration.json"]
