# A2C2 Windows Host Preparation

The 24 GB-class host initially used `21,165,830,144 / 24,871,014,400`
physical bytes (`85.1024%`), with `3,705,184,256` bytes available,
`36,840,108,032` committed bytes, zero page reads/writes, and `2,055 MiB`
current pagefile usage. No research worker, WSL VM, or game was active, and the
original `.wslconfig` state was absent.

The largest observed non-system families were Chrome (about 2.0 GiB aggregate)
and Cursor (about 1.5 GiB), followed by Claude and Riot client processes.
Chrome, Claude, Riot services, Settings, Widgets/unrelated WebViews, OneDrive,
GGQ, KakaoTalk, and extra Explorer windows were safely closed. Cursor, Codex,
and system/security processes were retained as essential. Reversible working
set trimming was used; no user data was deleted.

The four launch baselines were `69.288%`, `68.144%`, `66.741%`, and `68.256%`.
They did not reach the preferred 55% or acceptable 65% targets, but each was
below the hard 70% no-launch boundary. Pagefile activity was flat before every
launch. At completion the temporary `.wslconfig` was removed, WSL was shut
down, and no research worker remained.
