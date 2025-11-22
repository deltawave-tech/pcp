#!/usr/bin/env /bin/bash

LOG="build-$(date -Iseconds).log"

echo "# --------- #" > "${LOG}"
echo "# flake.nix #" > "${LOG}"
echo "# --------- #" > "${LOG}"
cat flake.nix > "${LOG}"

echo "# -------------- #"
echo "# Starting build #"
echo "# -------------- #"
nix build -L -v '.#iree-sdk' 2>&1 | tee "${LOG}"
