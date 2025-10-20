#!/bin/bash

OUTPUT_GIF="gnuplot_animation.gif"
GNUPLOT_SCRIPT="temp_anim_script.gp"

FPS=25
DELAY=$((100 / FPS))
WIDTH=640
HEIGHT=480

echo "set terminal gif animate optimize size ${WIDTH},${HEIGHT} delay $DELAY loop 0" > "$GNUPLOT_SCRIPT"
echo "set output '$OUTPUT_GIF'" >> "$GNUPLOT_SCRIPT"

ls -1 out_*.data | sed 's/out_//; s/\.data//' | sort -n | sed "s/^/plot 'out_/; s/$/\.data' with image/" >> "$GNUPLOT_SCRIPT"

echo "set output" >> "$GNUPLOT_SCRIPT"

gnuplot "$GNUPLOT_SCRIPT"

rm "$GNUPLOT_SCRIPT"

echo "Done. Animation saved as: $OUTPUT_GIF"