#!/bin/bash -l

convert -delay 150 -loop 0 *.png motion_diff.gif
convert motion_diff.gif -resize 20% motion_diff_compressed.gif
