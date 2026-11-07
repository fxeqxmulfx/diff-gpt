# Complete histories; no smoothing. Run after generating the two TSVs.
# default: SVG. gnuplot -e 'render_png=1' pow4-plain-ffn.gnuplot for PNG.
if (!exists("render_png")) render_png=0
set encoding utf8
set datafile separator "\t"
set border 3
set tics nomirror
set grid ytics lc rgb "#d9e0e8"
set key outside below horizontal
set xlabel "Epochs from the identical seed-1 initialization"
if (render_png) {
    set terminal pngcairo size 1200,1450 enhanced font "sans,13"
    set output "pow4-plain-ffn.png"
} else {
    set terminal svg size 1200,1450 dynamic enhanced font "sans,13"
    set output "pow4-plain-ffn.svg"
}
set multiplot layout 4,1 title "(a+b)^4 mod 97, seed 1: radial versus plain complex-product FFN\nSame initialization, samples, optimizer, norms, residuals and attention; first-99 stopping"
set ylabel "Held-out accuracy (%)"
set xrange [0:*]
set yrange [0:100]
set ytics 20
set arrow 1 from graph 0, first 99 to graph 1, first 99 nohead dt 2 lc rgb "#555555"
plot "pow4-plain-ffn-radial.tsv" using 2:3 with lines lw 1.2 lc rgb "#176ba5" title "Radial gate (audited control)", \
     "pow4-plain-ffn-plain.tsv" using 2:3 with lines lw 1.2 lc rgb "#c56028" title "Plain product"
set title "Upper accuracy range (same observations)"
set yrange [80:100]
set ytics 5
plot "pow4-plain-ffn-radial.tsv" using 2:3 with lines lw 1.2 lc rgb "#176ba5" title "Radial gate", \
     "pow4-plain-ffn-plain.tsv" using 2:3 with lines lw 1.2 lc rgb "#c56028" title "Plain product"
unset arrow 1
unset title
set ylabel "Full-training cross-entropy"
set logscale y
set yrange [*:*]
set ytics autofreq
set format y "10^{%L}"
plot "pow4-plain-ffn-radial.tsv" using 2:4 with lines lw 1.1 lc rgb "#176ba5" title "Radial gate", \
     "pow4-plain-ffn-plain.tsv" using 2:4 with lines lw 1.1 lc rgb "#c56028" title "Plain product"
unset logscale y
set format y "%g"
set ylabel "Plain-model a=b accuracy (%)"
set yrange [0:100]
set ytics 20
plot "pow4-plain-ffn-plain.tsv" using 2:6 with lines lw 1.2 lc rgb "#727b84" title "Training diagonals (55)", \
     "pow4-plain-ffn-plain.tsv" using 2:7 with lines lw 1.2 lc rgb "#c56028" title "Held-out diagonals (42)"
unset multiplot
unset output
