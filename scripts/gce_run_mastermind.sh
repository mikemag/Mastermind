#!/bin/bash
echo "Running Mastermind..."

export MASTERMIND_STATS_TAG=`hostname -s`
~/dev/Mastermind/mastermind

sudo poweroff

echo "Done."
