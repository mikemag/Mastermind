#!/bin/bash
echo "Copying all results to storage bucket..."
su "mike" -c "gsutil -m cp /home/mike/dev/Mastermind/mastermind*.json gs://mikemag-compute-bucket/mastermind/"
echo "Done."
