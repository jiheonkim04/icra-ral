# A2C2 Clean-Host Preflight

Decision: `A2C2_CLEAN_HOST_PREFLIGHT_ACCEPTED`

The current host still has `24,871,014,400` physical bytes. After reboot it
used `42.234%`; safely closing Chrome, OP.GG, and GGQ reduced use to `38.332%`
with `15,337,545,728` bytes available and `11,725,131,776` committed bytes.
Pagefile current use remained `34 MiB` and page writes were zero.

No old Python/WSL/LIBERO/MuJoCo/A2C2/X-VLA worker, game, or research GPU
compute process was active. Cursor, Codex, Explorer, and Windows security,
driver, and system processes were retained. The original `.wslconfig` state
was absent. The cleaned host passes the preferred `<=55%` launch target and
also satisfies the conditional 14 GB baseline requirement if the 8/10/12 GB
sequence later justifies it.
