#!/bin/sh
# Download Universal Dependencies English Web Treebank for sentence boundary benchmarking
# Source: https://github.com/UniversalDependencies/UD_English-EWT
set -e

OUTDIR="testdata/ud-ewt"
mkdir -p "$OUTDIR"

BASE_URL="https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"

echo "Downloading UD English Web Treebank..."

for split in train dev test; do
    file="en_ewt-ud-${split}.conllu"
    url="${BASE_URL}/${file}"
    outfile="${OUTDIR}/${file}"

    if [ -f "$outfile" ]; then
        echo "  ${file} already exists, skipping"
    else
        echo "  Downloading ${file}..."
        curl -sL "$url" -o "$outfile"
    fi
done

echo "Done. Run 'go run ./scripts/process-ud-ewt.go' to convert to benchmark format."
