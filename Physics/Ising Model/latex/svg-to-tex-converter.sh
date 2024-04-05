#!env bash

for file in ./figures/*.svg;
do
	filename=$(basename "${file}" ".svg")
	inkscape --export-latex -D "${file}" -o "./figures/${filename}.pdf"
done
