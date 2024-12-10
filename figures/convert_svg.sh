#!/bin/bash

declare -a SVG_MAIN_FILE=(
    models/sample                    # 1
    models/models                    # 2
	generalization/generalization    # 3
	motion/motion                    # 4
	motion/motion2                   # 5
	motion_self/motion_self_2        # 6
	convergence/convergence          # 7
	motion_self/motion_self_3        # 8
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