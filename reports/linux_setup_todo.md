# Linux / WSL Setup TODO

PowerShell preflight is the supported first path on Windows.

For real training, Linux or WSL is recommended because robotics simulators, CUDA wheels, and VLA dependencies are usually tested there first.

## Before real training

1. Install a working Linux or WSL environment.
2. Confirm NVIDIA GPU visibility inside that environment.
3. Install PyTorch with the CUDA version matching the driver.
4. Install simulator dependencies for LIBERO, RoboSuite, and any RoboCasa task suite.
5. Configure local asset paths in `configs/paths.local.yaml` or environment variables.
6. Run preflight again from Linux/WSL.
7. Run only dummy smoke first, then a real adapter smoke if all checks pass.

Do not run full GPU jobs until the tiny pilot go/no-go passes.
