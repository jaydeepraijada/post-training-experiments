#!/bin/bash
set -e

echo "Starting setup for Ubuntu cloud VM..."

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "Installing system packages..."
$SUDO apt-get update -y
$SUDO apt-get install -y tmux vim curl git

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv already installed."
fi

if [[ "$TERM" == "xterm-ghostty" ]]; then
    echo "export TERM=xterm-256color" >> ~/.bashrc
    export TERM=xterm-256color
fi

echo "Setup complete. Run: source \$HOME/.cargo/env"
