# Use a standard Ubuntu base image
FROM ubuntu:22.04

# Install prerequisites as root
RUN apt-get update && apt-get install -y --no-install-recommends curl git sudo xz-utils ca-certificates

# Create Nix config to disable seccomp syscall filtering (fixes BPF load error under emulation on Apple Silicon)
# Also enable flakes and nix-command by default
RUN mkdir -p /etc/nix && \
    echo "filter-syscalls = false" >> /etc/nix/nix.conf && \
    echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf

# Create a non-root user for development
RUN useradd -m -s /bin/bash dev

# Create /nix and chown to dev (required for single-user Nix install)
RUN mkdir -m 0755 /nix && chown dev /nix

# Switch to the new non-root user
USER dev

# Set USER env var explicitly (ensures it's available in shells and during install)
ENV USER=dev

# Install Nix in single-user mode (non-root) using the official installer non-interactively
RUN curl --proto '=https' --tlsv1.2 -sSf -L https://nixos.org/nix/install | sh -s -- --no-daemon --yes

# Add sourcing of nix.sh to .bashrc (for non-login interactive shells in Docker; with $USER set, it will modify PATH and env vars)
RUN echo 'source /home/dev/.nix-profile/etc/profile.d/nix.sh' >> /home/dev/.bashrc

# Start an interactive shell (will source .bashrc automatically)
CMD ["/bin/bash"]
