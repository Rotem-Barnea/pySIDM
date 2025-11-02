#!/bin/bash
USER="rotembarnea"
REMOTE_HOST="slurmlogin.tau.ac.il"
REMOTE_BASE="~/SIDM/pySIDM/run results/test run 1"
LOCAL_BASE="$HOME/Downloads/power_results"

REMOTE_END=$(basename "$REMOTE_BASE")

if [[ "$LOCAL_BASE" != *"$REMOTE_END" ]]; then
    LOCAL_BASE="$LOCAL_BASE/$REMOTE_END"
fi

mkdir -p "$LOCAL_BASE"

rsync -avz "${USER}@${REMOTE_HOST}:${REMOTE_BASE}/" "${LOCAL_BASE}/"
if [ $? -eq 0 ]; then
    echo "Download complete, files at $LOCAL_BASE"
else
    echo "Download failed. Check VPN or SSH connection."
fi
