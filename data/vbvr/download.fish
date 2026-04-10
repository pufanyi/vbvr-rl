#!/usr/bin/env fish

set -l LOCAL_DIR (status dirname)/VBVR-Dataset
set -l MAX_RETRIES 10
set -l RETRY_DELAY 30

for i in (seq $MAX_RETRIES)
    echo "[$i/$MAX_RETRIES] Downloading..."
    hf download Video-Reason/VBVR-Dataset --local-dir $LOCAL_DIR --repo-type dataset
    if test $status -eq 0
        echo "Downloaded to $LOCAL_DIR"
        exit 0
    end
    echo "Failed (attempt $i/$MAX_RETRIES), retrying in {$RETRY_DELAY}s..."
    sleep $RETRY_DELAY
end

echo "Failed after $MAX_RETRIES attempts."
exit 1
