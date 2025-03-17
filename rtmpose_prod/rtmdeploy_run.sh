#!/bin/bash

# Script usage function
usage() {
    echo "Usage: $0 [-v video_path] [-r results_path] [-i image_name] [-z] [-m model_type]"
    echo "Options:"
    echo "  -v : Path to videos directory (default: videos)"
    echo "  -r : Path to results directory (default: videos/res)"
    echo "  -i : Docker image name (default: rtmdeploy_prod:latest)"
    echo "  -z : Enable visualization"
    echo "  -m : Model type (coco, wholebody or halpe)"
    exit 1
}

# Default values
VIDEO_PATH="videos"
RESULTS_PATH="res"
IMAGE_NAME="rtmdeploy_prod:latest"
VIZ_FLAG=""
MODEL=""

# Parse command line arguments
while getopts "v:r:i:zm:h" opt; do
  case $opt in
    v) VIDEO_PATH="$OPTARG"
    ;;
    r) RESULTS_PATH="$OPTARG"
    ;;
    i) IMAGE_NAME="$OPTARG"
    ;;
    z) VIZ_FLAG="--visualize"
    ;;
    m) case "$OPTARG" in
         "coco") MODEL="rtmpose-x-b8";;
         "halpe") MODEL="rtmpose-x-b8-halpe";;
         "wholebody") MODEL="rtmw-x";;
         *) echo "Invalid model type. Use 'coco' or 'halpe'"; exit 1;;
       esac
    ;;
    h) usage
    ;;
    \?) echo "Invalid option -$OPTARG" >&2; usage
    ;;
  esac
done

# Convert relative path to absolute path
if [ "${VIDEO_PATH#/}" = "${VIDEO_PATH}" ]; then
    VIDEO_PATH="$(pwd)/${VIDEO_PATH}"
    echo "Converting to absolute path: ${VIDEO_PATH}"
fi

# Display current configuration
echo "Running with configuration:"
echo "Video folder: ${VIDEO_PATH}"

# Create results directory if not specified
# if [ -z "${RESULTS_PATH}" ]; then
#     RESULTS_PATH="${VIDEO_PATH}/res"
#     echo "No results path specified, creating: ${RESULTS_PATH}"
#     mkdir -p "${RESULTS_PATH}" || {
#         echo "Error: Could not create results directory ${RESULTS_PATH}"
#         exit 1
#     }
# fi




# RESULTS_PATH="${VIDEO_PATH}/res"

echo "Results folder: ${RESULTS_PATH}"
echo "Image name: ${IMAGE_NAME}"

docker run --gpus all --rm \
    -v "${VIDEO_PATH}":/data/videos \
    "${IMAGE_NAME}" ${VIZ_FLAG} --res_folder /data/videos/${RESULTS_PATH} --pose_path "models/${MODEL}"