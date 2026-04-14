# For TacEx docker image
ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base:2.1.1
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

SHELL ["/bin/bash", "-c"]

ARG ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV ISAACLAB_EXTENSION_TEMPLATE_PATH=${ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}
ARG ISAACLAB_PATH_ARG
ENV ISAACLAB_PATH=${ISAACLAB_PATH_ARG}
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ARG CMAKE_CUDA_ARCHITECTURES
ENV CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}

ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
USER root

# ── 1. System deps (rarely changes) ──────────────────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc-11 g++-11 python3.10-dev \
    git git-lfs wget ffmpeg nano tmux \
    pkg-config zip unzip ca-certificates \
    xvfb x11-utils && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# ── 2. CUDA paths ─────────────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

FROM nvcr.io/nvidia/cuda:12.6.0-devel-ubuntu22.04 AS cuda-devel

FROM base
COPY --from=cuda-devel /usr/local/cuda /usr/local/cuda

# ── 3. Aliases + pip upgrade (rarely changes) ─────────────────────────────────
RUN echo "alias run=/isaac-sim/isaac-sim.streaming.sh" >> ${HOME}/.bashrc && \
    echo "alias lab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc

RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    rm /usr/bin/python3.10 && \
    ln -s /isaac-sim/kit/python/bin/python3.10 /usr/bin/python3.10

# ── 4. CMake 3.26 (rarely changes) ───────────────────────────────────────────
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    chmod +x cmake-3.26.4-linux-x86_64.sh && \
    mkdir /opt/cmake && \
    ./cmake-3.26.4-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    rm cmake-3.26.4-linux-x86_64.sh
ENV PATH="/opt/cmake/bin:${PATH}"

# ── 5. vcpkg (rarely changes) ─────────────────────────────────────────────────
RUN cd /opt && \
    git clone https://github.com/microsoft/vcpkg.git && \
    cd vcpkg && ./bootstrap-vcpkg.sh -disableMetrics
ENV VCPKG_ROOT=/opt/vcpkg
ENV CMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake

ARG CACHEBUST=1
# ── 6. IsaacLab install (rarely changes) ──────────────────────────────────────
RUN /isaac-sim/python.sh -m pip install pip==23.* && \
    /isaac-sim/python.sh -m pip install setuptools==65.* && \
    /isaac-sim/python.sh -m pip install flatdict && \
    cd ${ISAACLAB_PATH} && ./isaaclab.sh --install
# ── 6. Copy only dependency sources needed for compilation ───────────────────
COPY third_party/TacEx/source ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/third_party/TacEx/source
COPY curobo ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/curobo

RUN bash -i -c "git config --global --add safe.directory /workspace/tacex"

# ── 7. TacEx core packages (fast) ────────────────────────────────────────────
RUN /isaac-sim/python.sh -m pip install -e \
    ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/third_party/TacEx/source/tacex && \
    /isaac-sim/python.sh -m pip install -e \
    ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/third_party/TacEx/source/tacex_assets && \
    /isaac-sim/python.sh -m pip install -e \
    ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/third_party/TacEx/source/tacex_tasks && \
    /isaac-sim/python.sh -m pip install -e \
    ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/third_party/TacEx/source/tacex_uipc -v

# ── 7b. Preserve libuipc runtime .so files outside the mounted volume ─────────
# The cmake build above writes libuipc*.so into the tacex_uipc source tree, which
# is bind-mounted at runtime and would therefore shadow the image-layer artifacts.
# Copy them to /opt/uipc_libs (not mounted) so they are always reachable.
RUN mkdir -p /opt/uipc_libs && \
    find ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/third_party/TacEx/source/tacex_uipc \
         -name "libuipc*.so*" -exec cp -P {} /opt/uipc_libs/ \;
ENV LD_LIBRARY_PATH=/opt/uipc_libs:${LD_LIBRARY_PATH}

# ── 8. curobo (slow compile) ──────────────────────────────────────────────────
RUN /isaac-sim/python.sh -m pip install wheel setuptools && \
    /isaac-sim/python.sh -m pip install \
    ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/curobo \
    --no-build-isolation && \
    /isaac-sim/python.sh -m pip install transforms3d
    

WORKDIR ${ISAACLAB_EXTENSION_TEMPLATE_PATH}
ENV HEADLESS=0