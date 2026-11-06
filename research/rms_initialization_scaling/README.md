# Initialization scale before finite-epsilon RMS

The user asks whether different model blocks need different initialization rules.
This [plan](plan.json) isolates a mathematical issue before choosing an initializer:
scaling a linear map immediately before RMS can change its parameter gradients
even when its normalized forward values barely move.

The existing Lean library already proves the exact identity
`R_eps(s*z) = R_(eps/s²)(z)` for positive `s`. The planned supplement lifts this
identity to a linear parameter map and its derivative, and studies radial versus
tangent sensitivity. It retains epsilon and fixed downstream inputs explicitly.
A declared 96-case CPU grid will compare actual RMS and matrix gradients with
independent formulas. No optimizer or task evaluation is involved.

Status: plan declared; proof and numerical check pending. The running capacity
grid and queued usual-gain experiment retain their frozen initializers.
