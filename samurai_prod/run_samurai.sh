#!/bin/bash

# Script usage function
usage() {
    echo "Usage: $0 [-v video_path]"
    echo "Options:"
    echo "  -v : Path to videos directory (default: videos)"

    exit 1
}

# Default values
VIDEO_PATH="videos"
IMAGE_NAME="samurai_prod:latest"

# Parse command line arguments
while getopts "v:h" opt; do
  case $opt in
    v) VIDEO_PATH="$OPTARG"
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

echo "Image name: ${IMAGE_NAME}"

docker run --gpus all --rm \
    -v "${VIDEO_PATH}":/data/videos \
    "${IMAGE_NAME}" 