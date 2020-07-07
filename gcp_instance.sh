#!/usr/bin/env bash

# Inspired by https://course.fast.ai/start_gcp.html#storage

set -euox pipefail

export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="europe-west1-b"
export INSTANCE_NAME="weakly-supervised-semseg"
export INSTANCE_TYPE="n1-highmem-8"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-p100,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-type=pd-ssd \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible

gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080

# Copy data
gcloud compute scp --recurse data/ jupyter@weakly-supervised-semseg:/home/jupyter/weakly-supervised-semseg/
gcloud compute scp data/ISPRS_semantic_labeling_Vaihingen/top_tiles.zip jupyter@weakly-supervised-semseg:/home/jupyter/weakly-supervised-semseg/data/ISPRS_semantic_labeling_Vaihingen/top_tiles.zip
gcloud compute scp data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_ADJ_tiles.zip jupyter@weakly-supervised-semseg:/home/jupyter/weakly-supervised-semseg/data/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_ADJ_tiles.zip
