#!/bin/sh
# Fetch Project Gutenberg books for benchmarking
# Usage: ./scripts/fetch-gutenberg.sh

set -e

OUTDIR="testdata/gutenberg"
mkdir -p "$OUTDIR"

download_book() {
    id="$1"
    name="$2"
    url="https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
    outfile="$OUTDIR/${name}_raw.txt"

    if [ -f "$outfile" ]; then
        echo "Skipping $name (already downloaded)"
        return
    fi

    echo "Downloading $name (ID: $id)..."
    curl -s -L -o "$outfile" "$url"
    sleep 2
}

download_book 1342 "pride_and_prejudice"
download_book 2701 "moby_dick"
download_book 1400 "great_expectations"
download_book 1228 "origin_of_species"
download_book 74 "tom_sawyer"
download_book 1260 "jane_eyre"

echo "Done. Run 'go run ./scripts/process-gutenberg.go' to process the files."
