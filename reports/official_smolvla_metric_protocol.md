# Official SmolVLA Metric Protocol

Date: 2026-07-10 KST

Primary metric: aggregate raw 7D action L2 after official SmolVLA postprocessing

Dimensions:

- translation: `[0, 1, 2]`
- rotation: `[3, 4, 5]`
- gripper: `[6]`

Secondary metrics:

- normalized SmolVLA eval loss
- translation L2 over action dims 0:3
- rotation L2 over action dims 3:6
- gripper absolute error on dim 6
- gripper sign accuracy on dim 6
- per-action-dimension absolute error and L2
- per-task breakdown
- per-episode breakdown
- action validity/range violation rate using official action stats
- help/hurt counts vs frozen/base
- win counts across fixed test tasks/subsets
- episode bootstrap confidence interval when cheap
- task bootstrap confidence interval when cheap

Averaging:

- primary: frame-weighted mean over fixed test frames
- required_secondary: task-balanced mean of per-task frame means
- episode_report: per-episode means and episode-bootstrap intervals
- task_report: per-task means and task-bootstrap intervals

Static alpha:

- grid: `[0.0, 0.25, 0.5, 0.75, 1.0]`
- selection: `choose alpha on validation split only by primary action L2, then freeze for test`
- test_tuning_allowed: `False`

Oracle baselines are upper bounds only and must not be presented as realistic methods.
