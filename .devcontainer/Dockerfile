FROM jfxs/devcontainer-rye

USER vscode

# Ensure .cache directory exists and has correct permissions
RUN mkdir -p /home/vscode/.cache \
    && sudo chown -R vscode:vscode /home/vscode/.cache

# Ensure .rye directory has correct permissions
RUN sudo chown -R vscode:vscode /home/vscode/.rye

# Install aider-chat
RUN rye install aider-chat

WORKDIR /workspaces