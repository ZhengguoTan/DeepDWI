#!/bin/bash

declare -a SVG_MAIN_FILE=(
    models/model_vae                 # 1
	regularizations/regularizations  # 2
	motion/motion                    # 3
	motion/motion2                   # 4
	motion_self/motion_self          # 5
)

ARR_LEN=${#SVG_MAIN_FILE[@]}

for (( N=0; N<${ARR_LEN}; N++ )); do

    echo "  converting Figure $((N+1))"

	PDF_DST=fig$((N+1))
	PNG_DST=fig$((N+1))

	# inkscape --version < 1
	# inkscape --export-pdf=${PDF_DST}.pdf -d 300 ${SVG_MAIN_FILE[$N]}.svg
	# inkscape --export-png=${PNG_DST}.png -d 300 ${SVG_MAIN_FILE[$N]}.svg

	# inkscape --version > 1
	inkscape --export-type="png" --export-filename=${PNG_DST} -d 300 ${SVG_MAIN_FILE[$N]}.svg

	convert ${PNG_DST}.png ${PNG_DST}.tiff
done