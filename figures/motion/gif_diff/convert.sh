#!/bin/bash -l

convert -delay 150 -loop 0 *.png motion_diff.gif
convert motion_diff.gif -resize 19% motion_diff_compressed.gif
