# PTC-VLA Prototype Protocol

Date: 2026-07-12 KST

1. Implement `tca_map/smolvla/ptc_vla.py`.
2. Implement `scripts/run_ptc_vla_prototype.py`.
3. Add focused tests in `tests/test_ptc_vla.py`.
4. Run synthetic mechanism smoke.
5. Run real trace training on official SmolVLA/LIBERO training identities.
6. Run Stage A if synthetic and trace training pass.
7. Apply the preregistered Stage A decision rules.

Expected commands:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_ptc_vla.py
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_ptc_vla_prototype.py --mode synthetic
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_ptc_vla_prototype.py --mode real-trace-train
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_ptc_vla_prototype.py --mode stage-a
```
