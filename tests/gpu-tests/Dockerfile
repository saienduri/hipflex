# GPU Tests — Self-contained image for hipflex GPU conformance tests.
#
# Downloads ROCm via TheRock's install script, builds hipflex from source,
# and runs pytest-based GPU tests. No volume mounts needed.
#
# Build:
#   docker build -t hipflex-tests -f tests/gpu-tests/Dockerfile .
#
# Build with specific ROCm version:
#   docker build -t hipflex-tests -f tests/gpu-tests/Dockerfile \
#     --build-arg ROCM_VERSION=7.12.0rc0 \
#     --build-arg ROCM_RELEASE_TYPE=prereleases .
#
# Run (requires AMD GPU passthrough):
#   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video hipflex-tests
#
# Run specific test:
#   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
#     hipflex-tests pytest -v tests/gpu-tests/test_standalone_mode.py

# ── Stage 1: Build hipflex ──

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    libssl-dev \
    llvm-dev \
    libclang-dev \
    clang \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /workspace
WORKDIR /workspace

RUN cargo build --release -p hipflex

# ── Stage 2: Runtime ──

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    curl \
    libatomic1 \
    libnuma1 \
    libdrm2 \
    libelf1 \
    && rm -rf /var/lib/apt/lists/*

# Install ROCm via TheRock's install script
ARG ROCM_VERSION=7.12.0rc0
ARG ROCM_AMDGPU_FAMILY=gfx94X-dcgpu
ARG ROCM_RELEASE_TYPE=prereleases

RUN curl -fsSL -o /tmp/install_rocm.sh \
    https://raw.githubusercontent.com/ROCm/TheRock/main/dockerfiles/install_rocm_tarball.sh && \
    chmod +x /tmp/install_rocm.sh && \
    /tmp/install_rocm.sh "${ROCM_VERSION}" "${ROCM_AMDGPU_FAMILY}" "${ROCM_RELEASE_TYPE}" && \
    rm -f /tmp/install_rocm.sh

ENV ROCM_PATH=/opt/rocm
ENV PATH="${ROCM_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib"

RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf && ldconfig

# Copy built library from builder stage
COPY --from=builder /workspace/target/release/libhipflex.so /workspace/libhipflex.so
ENV HIPFLEX_LIB=/workspace/libhipflex.so

# Copy test files
COPY tests/gpu-tests/ /workspace/tests/gpu-tests/

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir pytest>=7.0

WORKDIR /workspace
CMD ["pytest", "-v", "tests/gpu-tests/"]
