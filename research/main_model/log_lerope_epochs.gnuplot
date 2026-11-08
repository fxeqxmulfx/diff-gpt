# Run only after all30 long cells pass the independent audit.
# Arguments: TSV, output basename, minimum axis, maximum axis, logarithmic flag.
set datafile separator '\t'
set border 3
set tics nomirror
if (ARG5 + 0 == 1) { set logscale xy }
set xrange [ARG3+0:ARG4+0]
set yrange [ARG3+0:ARG4+0]
set size ratio -1
set grid back lc rgb '#d9dee7'
set key left top opaque box
set xlabel 'Selected LeRoPE main: epochs to first observed 99%'
set ylabel 'Log-Scaled LeRoPE candidate: epochs to first observed 99%'
set title 'Long benchmark: all30 task/seed pairs, independently audited'
set label 1 'Below the diagonal: candidate used fewer epochs' at graph 0.98,0.04 right front
set style line 1 lc rgb '#9ca3af' lw 1.5 dt 2
set terminal svg size 1000,900 dynamic enhanced font 'sans,13'
set output ARG2.'.svg'
plot x with lines ls 1 title 'Equal epochs', \
  for [seed=0:2] ARG1 every ::1 using \
    ((strcol(3) eq 'passed' && column(2)==seed)?column(4):1/0):5 \
    with points pt (7+seed*2) ps 1.2 lc rgb word('#2563eb #ea580c #16a34a',seed+1) title sprintf('Seed %d',seed)
set terminal pngcairo size 1200,1080 enhanced font 'sans,15'
set output ARG2.'.png'
replot
unset output
