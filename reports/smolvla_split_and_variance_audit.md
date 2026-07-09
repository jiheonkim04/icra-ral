# SmolVLA Split And Variance Audit

- previous 9/6 explanation: The prior runner sampled 3 train demos and 2 eval demos with 3 timesteps per demo, so records were sampled timestep/action-window records: 3*3 train and 2*3 eval.
- records are: sampled observation/action-chunk windows; each record is one demo/timestep plus a 50-step action chunk target
- raw demo count: `50`
- raw timestep count: `13298`
- global action variance: `[0.068752, 0.095218, 0.137241, 0.00123, 0.003477, 0.010566, 0.968331]`
- translation variance mean: `0.100404`
- rotation variance mean: `0.005091`
- gripper variance: `0.96829`
- previous split mean-action L2: `0.486561`
- previous split previous-action L2: `0.188748`
- larger demo split train/eval: `300 / 100`
- same-demo time split train/eval: `80 / 40`
- task holdout feasible without download: `True`
