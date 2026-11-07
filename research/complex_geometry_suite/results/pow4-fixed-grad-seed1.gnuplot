# Run from this directory after generating both complete history TSVs.
# All observations are plotted; no smoothing or observation selection.
set encoding utf8
set datafile separator "\t"
set border 3
set tics nomirror
set grid ytics lc rgb "#d9e0e8"
set xlabel "Epochs from the identical seed-1 initialization"
set ylabel "Complete held-out accuracy (%)"
set key outside below horizontal
set xrange [0:*]
set terminal svg size 1100,900 dynamic enhanced font "sans,14"
set output "pow4-fixed-grad-seed1.svg"
set multiplot layout 2,1 title "(a+b)^4 mod 97, seed 1: original vs corrected gradients\nIdentical initialization, samples, model and optimizer; first-99 stopping"
set yrange [0:100]
set ytics 20
set arrow 1 from graph 0, first 99 to graph 1, first 99 nohead dt 2 lc rgb "#555555"
plot "pow4-fixed-grad-seed1-original.tsv" using 2:4 with lines lw 1.2 lc rgb "#bf6328" title "Original", \
     "pow4-fixed-grad-seed1-corrected.tsv" using 2:4 with lines lw 1.4 lc rgb "#176ba5" title "Corrected loss + XSA adjoint"
set title "Upper accuracy range (same observations)"
set yrange [80:100]
set ytics 5
plot "pow4-fixed-grad-seed1-original.tsv" using 2:4 with lines lw 1.2 lc rgb "#bf6328" title "Original", \
     "pow4-fixed-grad-seed1-corrected.tsv" using 2:4 with lines lw 1.4 lc rgb "#176ba5" title "Corrected loss + XSA adjoint"
unset multiplot
unset output
