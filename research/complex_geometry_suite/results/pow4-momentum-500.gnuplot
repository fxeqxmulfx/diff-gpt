# Run from this directory. Set render_png=1 for PNG; default is SVG.
# Complete unsmoothed histories, including every unfavorable observation.
if (!exists("render_png")) render_png=0
set encoding utf8
set datafile separator "\t"
set border 3
set tics nomirror
set grid ytics lc rgb "#d9e0e8"
set xlabel "Epochs from the original initialization"
set xrange [4000:4500]
set key outside below horizontal
if (render_png) {
    set terminal pngcairo size 1200,1250 enhanced font "sans,13"
    set output "pow4-momentum-500.png"
} else {
    set terminal svg size 1200,1250 dynamic enhanced font "sans,13"
    set output "pow4-momentum-500.svg"
}
set multiplot layout 3,1 title "(a+b)^4 mod 97, seed 1: a one-time first-moment reset\nIdentical epoch-4000 weights, LR .002, second moment and next 500 epochs of batches"
set ylabel "Held-out accuracy (%)"
set yrange [80:100]
set ytics 5
set arrow 1 from graph 0, first 99 to graph 1, first 99 nohead dt 2 lc rgb "#555555"
plot "pow4-momentum-500-keep.tsv" using 2:3 with lines lw 1.4 lc rgb "#176ba5" title "Keep AdamW moments", \
     "pow4-momentum-500-reset_first.tsv" using 2:3 with lines lw 1.4 lc rgb "#c56028" title "Reset first moment once"
unset arrow 1
set ylabel "Full-training cross-entropy"
set logscale y
set yrange [*:*]
set ytics autofreq
set format y "10^{%L}"
plot "pow4-momentum-500-keep.tsv" using 2:4 with lines lw 1.3 lc rgb "#176ba5" title "Keep AdamW moments", \
     "pow4-momentum-500-reset_first.tsv" using 2:4 with lines lw 1.3 lc rgb "#c56028" title "Reset first moment once"
unset logscale y
set format y "%g"
set ylabel "Held-out a=b correct (out of 42)"
set yrange [0:42]
set ytics 7
plot "pow4-momentum-500-keep.tsv" using 2:6 with lines lw 1.2 lc rgb "#176ba5" title "Keep AdamW moments", \
     "pow4-momentum-500-reset_first.tsv" using 2:6 with lines lw 1.2 lc rgb "#c56028" title "Reset first moment once"
unset multiplot
unset output
