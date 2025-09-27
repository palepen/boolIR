#!/bin/bash
#
# This script downloads the necessary TREC-6 evaluation files:
# 1. Topics (Queries) 301-350
# 2. Qrels (Relevance Judgments) for the ad-hoc task

# --- Configuration ---
DATA_DIR="data"
# CORRECTED URL for the gzipped topics file
TOPICS_URL="https://trec.nist.gov/data/topics_eng/topics.301-350.gz"
QRELS_URL="https://trec.nist.gov/data/qrels_eng/qrels.trec6.adhoc.gz"

TOPICS_GZ_FILE_NAME="topics.301-350.gz"
TOPICS_FILE_NAME="topics.301-350.txt"
QRELS_GZ_FILE_NAME="qrels.trec6.adhoc.gz"
QRELS_FILE_NAME="qrels.trec6.adhoc.txt"

# --- Script Logic ---
echo "▶️ Starting TREC-6 data download..."

# Create the data directory if it doesn't exist
mkdir -p "$DATA_DIR"
echo "✅ Ensured '$DATA_DIR' directory exists."

# Download and decompress the topics file if it's not already there
if [ ! -f "$DATA_DIR/$TOPICS_FILE_NAME" ]; then
    echo "Downloading topics file..."
    wget -O "$DATA_DIR/$TOPICS_GZ_FILE_NAME" "$TOPICS_URL"

    echo "Decompressing topics file..."
    gunzip -c "$DATA_DIR/$TOPICS_GZ_FILE_NAME" > "$DATA_DIR/$TOPICS_FILE_NAME"
    
    # Clean up the compressed file
    rm "$DATA_DIR/$TOPICS_GZ_FILE_NAME"
    echo "✅ Topics file downloaded and decompressed."
else
    echo "⏩ Topics file already exists. Skipping download."
fi

# Download and decompress the qrels file if it's not already there
if [ ! -f "$DATA_DIR/$QRELS_FILE_NAME" ]; then
    echo "Downloading qrels file..."
    wget -O "$DATA_DIR/$QRELS_GZ_FILE_NAME" "$QRELS_URL"

    echo "Decompressing qrels file..."
    gunzip -c "$DATA_DIR/$QRELS_GZ_FILE_NAME" > "$DATA_DIR/$QRELS_FILE_NAME"
    
    # Clean up the compressed file
    rm "$DATA_DIR/$QRELS_GZ_FILE_NAME"
    echo "✅ Qrels file downloaded and decompressed."
else
    echo "⏩ Qrels file already exists. Skipping download."
fi

echo "🎉 All evaluation files are ready in the '$DATA_DIR' directory."