#!/bin/bash
# Script to download and prepare the Cranfield dataset.

# Set the download URL and the target directory
DATASET_URL="http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz"
DOCS_DIR="docs"
OUTPUT_FILE="$DOCS_DIR/cran.tar.gz"
EXTRACT_DIR="$DOCS_DIR/cranfield_temp" # Use a temporary directory for extraction

# Create the docs directory if it doesn't exist
mkdir -p $DOCS_DIR

echo "Downloading Cranfield dataset..."

# Download the file
if command -v wget &> /dev/null; then
    wget -q -O $OUTPUT_FILE $DATASET_URL
elif command -v curl &> /dev/null; then
    curl -s -o $OUTPUT_FILE -L $DATASET_URL
else
    echo "Error: Neither wget nor curl is available."
    exit 1
fi

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Failed to download the dataset."
    exit 1
fi

echo "Extracting dataset..."
mkdir -p $EXTRACT_DIR
# **THE FIX:** Removed --strip-components=1 as it was incorrect for this archive
tar -xzf $OUTPUT_FILE -C $EXTRACT_DIR

# Check if the required file exists before processing
if [ ! -f "$EXTRACT_DIR/cran.all.1400" ]; then
    echo "Error: cran.all.1400 not found after extraction."
    exit 1
fi

echo "Processing documents..."
# Process the raw cran.all.1400 file into individual documents
awk '
BEGIN { doc_id = 0; content = ""; }
/^\.I/ {
    if (doc_id > 0) {
        # Save the previous document
        print content > "'$DOCS_DIR'/doc" doc_id ".txt";
    }
    doc_id++;
    content = "";
    next;
}
/^\.T/ || /^\.A/ || /^\.B/ || /^\.W/ {
    # Skip the field markers themselves
    next;
}
{
    # Append the line to the current document content
    content = content $0 " ";
}
END {
    # Save the last document
    if (doc_id > 0) {
        print content > "'$DOCS_DIR'/doc" doc_id ".txt";
    }
}' $EXTRACT_DIR/cran.all.1400


# Clean up intermediate files
rm -rf $EXTRACT_DIR
rm $OUTPUT_FILE

echo "✅ Dataset is ready in the '$DOCS_DIR' directory."