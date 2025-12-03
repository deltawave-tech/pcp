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
date >> "${LOG}"
nix build -L -v '.#pcp' --keep-failed 2>&1 | tee --append "${LOG}"
date >> "${LOG}"
