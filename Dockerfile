# Can try changing devel to base if you don't need to compile anything
ARG BASE_IMAGE=nvidia/cuda:12.6.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgl1 \
    libxcb-cursor0 \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set Pixi root to a fast local directory inside the container
ENV PIXI_HOME=/opt/pixi
ENV PATH=$PIXI_HOME/bin:$PIXI_HOME/envs/test/bin:$PATH

RUN rm -rf /root/.pixi

# Create the directory
RUN mkdir -p /opt/pixi && chmod -R 777 /opt/pixi


RUN curl -fsSL https://pixi.sh/install.sh | bash
WORKDIR /workspace
COPY . /workspace/

RUN rm -rf /tmp/* /var/tmp/*

# Use default environment
RUN pixi shell-hook -e default > /workspace/shell-hook.sh
RUN echo 'exec "$@"' >> /workspace/shell-hook.sh
RUN chmod +x /workspace/shell-hook.sh

# For Genesis only
RUN apt-get update && apt install -y libosmesa6 libosmesa6-dev && rm -rf /var/lib/apt/lists/*