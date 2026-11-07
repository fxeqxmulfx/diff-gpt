# Full unsmoothed histories. The candidate stops at first99 or100000updates.
if (!exists("render_png")) render_png=0
set encoding utf8
set datafile separator "\t"
set border 3
set tics nomirror
set grid ytics lc rgb "#d9e0e8"
set key outside below horizontal
stats "pow4-no-xsa-ordinary.tsv" using 2 nooutput
candidate_end=STATS_max
if (render_png) {
    set terminal pngcairo size 1200,1450 enhanced font "sans,13"
    set output "pow4-no-xsa.png"
} else {
    set terminal svg size 1200,1450 dynamic enhanced font "sans,13"
    set output "pow4-no-xsa.svg"
}
set multiplot layout 4,1 title "(a+b)^4 mod97, seed1: Hermitian XSA versus ordinary attention\nPlain complex product; same initialization, samples, optimizer and norms"
set xlabel "Epochs from identical initialization"
set ylabel "Held-out accuracy (%)"
set xrange [0:*]
set yrange [0:100]
set ytics 20
set arrow 1 from graph 0, first 99 to graph 1, first 99 nohead dt 2 lc rgb "#555555"
set arrow 2 from first candidate_end, graph 0 to first candidate_end, graph 1 nohead dt 3 lc rgb "#8a8a8a"
set label 1 "Ordinary model stops at first 99%" at first candidate_end+90, first 60 textcolor rgb "#555555"
plot "pow4-no-xsa-xsa.tsv" using 2:3 with lines lw 1.2 lc rgb "#176ba5" title "Hermitian XSA (all observed)", \
     "pow4-no-xsa-ordinary.tsv" using 2:3 with lines lw 1.2 lc rgb "#c56028" title "Ordinary attention (all observed)"
unset arrow 2
unset label 1
set title "Matched exposure, upper accuracy range"
set xrange [0:candidate_end]
set yrange [80:100]
set ytics 5
plot "pow4-no-xsa-xsa.tsv" using 2:3 with lines lw 1.2 lc rgb "#176ba5" title "Hermitian XSA", \
     "pow4-no-xsa-ordinary.tsv" using 2:3 with lines lw 1.2 lc rgb "#c56028" title "Ordinary attention"
unset arrow 1
unset title
set ylabel "Full-training cross-entropy"
set logscale y
set yrange [*:*]
set ytics autofreq
set format y "10^{%L}"
plot "pow4-no-xsa-xsa.tsv" using 2:4 with lines lw 1.1 lc rgb "#176ba5" title "Hermitian XSA", \
     "pow4-no-xsa-ordinary.tsv" using 2:4 with lines lw 1.1 lc rgb "#c56028" title "Ordinary attention"
unset logscale y
set format y "%g"
set ylabel "Held-out a=b accuracy (%)"
set yrange [0:100]
set ytics 20
plot "pow4-no-xsa-xsa.tsv" using 2:7 with lines lw 1.1 lc rgb "#176ba5" title "Hermitian XSA diagonals (42)", \
     "pow4-no-xsa-ordinary.tsv" using 2:7 with lines lw 1.1 lc rgb "#c56028" title "Ordinary attention diagonals (42)"
unset multiplot
unset output
