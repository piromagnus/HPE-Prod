#!/bin/bash

# Create models directory if it doesn't exist
MODELS_DIR=".."
mkdir -p "$MODELS_DIR"

# URL for the models
URL="https://cloud2-ljk.imag.fr/index.php/s/TJMT8ecoy769EyH/download"

# Download and extract the models
echo "Downloading models..."
wget -O "$MODELS_DIR/models.zip" "$URL"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed. Extracting files..."
    cd "$MODELS_DIR"
    unzip models.zip
    rm models.zip
    echo "Models have been downloaded and extracted successfully."
else
    echo "Error: Download failed!"
    exit 1
fi