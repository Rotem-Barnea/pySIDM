#!/bin/bash
USER="rotembarnea"
REMOTE_HOST="slurmlogin.tau.ac.il"
REMOTE_BASE="/scratch200/rotembarnea/SIDM/pySIDM/run results/run_7590998"
LOCAL_BASE="$HOME/Documents/SIDM/pySIDM/run results"

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
