#!/bin/bash

# Convert all SVG files in the current directory to PDF+LaTeX
find . -maxdepth 1 -type f -name "*.svg" -exec sh -c '
  for file; do
    filename="${file%.*}"
    org.inkscape.Inkscape -D "$file" -o "${filename}.pdf" --export-latex
  done
' sh {} +
