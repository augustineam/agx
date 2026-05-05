# Stage 1: Base Image
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Install additional dependencies
RUN apt update && apt install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    graphviz \
    sudo && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

# Change ubuntu user's primary group to 998 and add to other groups
RUN usermod -g 998 -aG users,sudo ubuntu
# Change password for user 'ubuntu'
RUN echo 'ubuntu:password' | chpasswd

# Switch to user ubuntu
USER ubuntu
WORKDIR /agx

# Expose MLflow UI port
EXPOSE 5000

# Start MLflow UI server
CMD ["tail", "-f", "/dev/null"]