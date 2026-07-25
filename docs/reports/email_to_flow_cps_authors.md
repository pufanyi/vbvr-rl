# Email Draft to the Flow-CPS Authors

**Subject:** Question about Flow-CPS training and ODE sampling performance

Dear Flow-CPS authors,

We have been experimenting with Flow-CPS for video generation and observed an
interesting sampler-dependent performance gap.

Using the same trained checkpoint, our current VBVR-Pro results are
approximately:

- Flow-CPS, $\eta=0.7$: **0.534**
- Flow-CPS, $\eta=0$: **0.510**
- FlowMatch Euler ODE: **0.505**
- UniPC ODE: **0.504**

The advantage of $\eta=0.7$ is also positive under task-level bootstrap. We
understand that our ODE and CPS pipelines are not yet perfectly matched in CFG,
timestep grid, and implementation details, so we do not regard this as a strict
causal comparison.

We would be grateful for your thoughts on the following questions:

1. Is it expected that a model trained through the stochastic CPS sampling path
   performs noticeably better with CPS sampling than with the probability-flow
   ODE?
2. Under identical timesteps, CFG, initial noise, and model evaluations, should
   the $\eta=0$ CPS implementation be expected to match first-order Euler
   exactly?

We also noticed that the log-probability expression in the arXiv v1 manuscript
appears to use a positive squared-error sign, whereas a Gaussian
log-probability and our implementation use a negative sign. Could you confirm
whether this is a typographical error?

We would be happy to share our implementation and a more detailed experimental
report if helpful.

Best regards,  
[Your Name]
