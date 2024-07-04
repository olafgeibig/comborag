FROM jfxs/devcontainer-rye

USER vscode

# Install necessary tools including a C++ compiler
RUN sudo apt-get update && sudo apt-get install -y \
    g++ \
    clang \
    make \
    cmake \
    libomp-dev \
    && sudo rm -rf /var/lib/apt/lists/*

# Ensure .cache directory exists and has correct permissions
RUN mkdir -p /home/vscode/.cache \
    && sudo chown -R vscode:vscode /home/vscode/.cache

# Ensure .rye directory has correct permissions
RUN sudo chown -R vscode:vscode /home/vscode/.rye

# Install aider-chat
RUN rye install aider-chat

WORKDIR /workspaces