#!/bin/bash

declare -a SVG_MAIN_FILE=(
    models/models                    # 1
	motion/motion                    # 2
	motion/motion2                   # 3
	motion_self/motion_self_2        # 4
	convergence/convergence          # 5
	motion_self/motion_self_3        # 6
	generalization/generalization    # 7
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