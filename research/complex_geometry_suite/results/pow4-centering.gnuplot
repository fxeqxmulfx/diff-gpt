# Full unsmoothed observations. Neither stopped trajectory is extrapolated.
if (!exists("render_png")) render_png=0
set encoding utf8
set datafile separator "\t"
set border 3
set tics nomirror
set grid ytics lc rgb "#d9e0e8"
set key outside below horizontal
stats "pow4-centering-ordinary.tsv" using 2 nooutput
ordinary_end=STATS_max
stats "pow4-centering-centered.tsv" using 2 nooutput
centered_end=STATS_max
common_end=(ordinary_end<centered_end ? ordinary_end : centered_end)
if (render_png) {
    set terminal pngcairo size 1200,1450 enhanced font "sans,13"
    set output "pow4-centering.png"
} else {
    set terminal svg size 1200,1450 dynamic enhanced font "sans,13"
    set output "pow4-centering.svg"
}
set multiplot layout 4,1 title "(a+b)^4 mod97, seed1: ordinary versus causal-centered attention\nPlain complex product; same initialization, samples, optimizer and norms"
set xlabel "Epochs from identical initialization"
set ylabel "Held-out accuracy (%)"
set xrange [0:*]
set yrange [0:100]
set ytics 20
set arrow 1 from graph 0, first 99 to graph 1, first 99 nohead dt 2 lc rgb "#555555"
set title "All observations; ordinary control ends at its first 99% crossing"
plot "pow4-centering-ordinary.tsv" using 2:3 with lines lw 1.2 lc rgb "#176ba5" title "Ordinary", \
     "pow4-centering-centered.tsv" using 2:3 with lines lw 1.2 lc rgb "#c56028" title "Causal centering"
set title "Common observed exposure"
set xrange [0:common_end]
set yrange [0:100]
set ytics 20
plot "pow4-centering-ordinary.tsv" using 2:3 with lines lw 1.2 lc rgb "#176ba5" title "Ordinary", \
     "pow4-centering-centered.tsv" using 2:3 with lines lw 1.2 lc rgb "#c56028" title "Causal centering"
unset arrow 1
unset title
set ylabel "Full-training cross-entropy"
set logscale y
set yrange [*:*]
set ytics autofreq
set format y "10^{%L}"
plot "pow4-centering-ordinary.tsv" using 2:4 with lines lw 1.1 lc rgb "#176ba5" title "Ordinary", \
     "pow4-centering-centered.tsv" using 2:4 with lines lw 1.1 lc rgb "#c56028" title "Causal centering"
unset logscale y
set format y "%g"
set ylabel "Held-out a=b accuracy (%)"
set yrange [0:100]
set ytics 20
plot "pow4-centering-ordinary.tsv" using 2:7 with lines lw 1.1 lc rgb "#176ba5" title "Ordinary diagonals (42)", \
     "pow4-centering-centered.tsv" using 2:7 with lines lw 1.1 lc rgb "#c56028" title "Centered diagonals (42)"
unset multiplot
unset output
