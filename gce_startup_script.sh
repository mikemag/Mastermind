#!/bin/bash
echo "Starting Mastermind..."
su mike -c "cd /home/mike/dev/Mastermind; /home/mike/dev/Mastermind/gce_run_mastermind_nohup.sh"
echo "Done."
