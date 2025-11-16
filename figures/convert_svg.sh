#!/bin/bash

declare -a SVG_MAIN_FILE=(
    models/sample                    # 1
    models/models                    # 2
    gt/gt                            # 3
    generalization/generalization    # 4
    navi/navi-self-gated             # 5
    navi/highres                     # 6
    motion/motion2                   # 7
    dti/dti                          # 8
    convergence/convergence          # 9
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


declare -a SVG_SUPP_FILE=(
    models/resnet                    # 1
    ablation/ablation                # 2
    intersubject/intersubject        # 3
)

ARR_LEN=${#SVG_SUPP_FILE[@]}

for (( N=0; N<${ARR_LEN}; N++ )); do

    echo "  converting Figure S$((N+1))"

    PDF_DST=figS$((N+1))
    PNG_DST=figS$((N+1))

    # inkscape --version < 1
    # inkscape --export-pdf=${PDF_DST}.pdf -d 300 ${SVG_MAIN_FILE[$N]}.svg
    # inkscape --export-png=${PNG_DST}.png -d 300 ${SVG_MAIN_FILE[$N]}.svg

    # inkscape --version > 1
    inkscape --export-type="png" --export-filename=${PNG_DST} -d 300 ${SVG_SUPP_FILE[$N]}.svg

    convert ${PNG_DST}.png ${PNG_DST}.tiff
done