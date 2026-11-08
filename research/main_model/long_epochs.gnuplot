# Standalone scientific figure; all 30 outcomes remain in the adjacent TSV.
set datafile separator '\t'
set border 3
set tics nomirror
set logscale xy
set xrange [min_epochs:max_epochs]
set yrange [min_epochs:max_epochs]
set size ratio -1
set grid back lc rgb '#d9dee7'
set key left top opaque box
set xlabel 'Main model: epochs to first observed 99%'
set ylabel 'mHC: epochs to first observed 99%'
set title sprintf('Long benchmark: %d / 30 pairs both reached 99%%', paired)
set label 1 "Below the diagonal: mHC used fewer epochs\nBudget misses are listed in mhc-long-epochs.tsv" at graph 0.98,0.06 right front
set style line 1 lc rgb '#9ca3af' lw 1.5 dt 2
set terminal svg size 950,850 dynamic enhanced font 'sans,13'
set output 'research/main_model/results/mhc-long-epochs.svg'
plot x with lines ls 1 title 'Equal epochs', \
  for [seed=0:2] 'research/main_model/results/mhc-long-epochs.tsv' every ::1 using \
    ((strcol(3) eq 'passed' && strcol(8) eq 'passed' && column(2)==seed)?column(4):1/0):9 \
    with points pt (7+seed*2) ps 1.2 lc rgb word('#2563eb #ea580c #16a34a',seed+1) title sprintf('Seed %d',seed)
set terminal pngcairo size 1140,1020 enhanced font 'sans,15'
set output 'research/main_model/results/mhc-long-epochs.png'
replot
unset output
