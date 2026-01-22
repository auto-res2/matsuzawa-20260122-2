import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    if cfg.run is None:
        raise ValueError("run=<run_id> must be specified")
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")

    script = Path(get_original_cwd()) / "src" / "train.py"
    cmd = [
        sys.executable,
        "-u",
        str(script),
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
