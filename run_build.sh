#!/usr/bin/env /bin/bash

LOG="build-$(date -Iseconds).log"

echo "############################################################" > "${LOG}"
echo "# --------- #" >> "${LOG}"
echo "# flake.nix #" >> "${LOG}"
echo "# --------- #" >> "${LOG}"
cat flake.nix >> "${LOG}"

echo "############################################################" >> "${LOG}"
echo "# -------------- #" >> "${LOG}"
echo "# Starting build #" >> "${LOG}"
echo "# -------------- #" >> "${LOG}"
nix build -L -v '.#pcp' 2>&1 | tee --append "${LOG}"
