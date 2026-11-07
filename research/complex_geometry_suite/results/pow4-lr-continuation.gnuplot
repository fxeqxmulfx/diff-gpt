# Run from this directory. No smoothing or omitted observations.
set encoding utf8
set datafile separator "\t"
set border 3
set tics nomirror
set grid ytics lc rgb "#d9e0e8"
set xrange [0:1000]
set yrange [75:100]
set ytics 5
set xlabel "Additional epochs from the same checkpoint"
set ylabel "Complete held-out accuracy (%)"
set title "Fourth-power arithmetic, Hermitian seed 1: matched continuation\nSame weights, moments and next 1,000 epochs of samples"
set key outside below horizontal
set arrow 1 from graph 0, first 99 to graph 1, first 99 nohead dt 2 lc rgb "#555555"
set label 1 "99% threshold" at graph 0.99, first 99.25 right tc rgb "#555555"
set terminal svg size 1100,620 dynamic enhanced font "sans,15"
set output "pow4-lr-continuation.svg"
plot "pow4-lr-continuation.tsv" using 1:4 with lines lw 1.3 lc rgb "#d56b24" title "LR 0.002", \
     "pow4-lr-continuation.tsv" using 1:6 with lines lw 2.2 lc rgb "#176ba5" title "LR 0.0002"
set terminal pngcairo size 1650,930 enhanced font "sans,19"
set output "pow4-lr-continuation.png"
replot
unset output
