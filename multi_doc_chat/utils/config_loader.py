from pathlib import Path
import os
import yaml

def _project_root() -> Path:
    """
    Returns the absolute path to the project's root directory.

    Assumes the root is two directories up from the current file
    (e.g., from 'project_root/utils/config_loader.py').

    Returns:
        Path: The absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]

def load_config(config_path: str | None = None) -> dict:
    """
    Loads configuration settings from a YAML file.

    The configuration file path is resolved in the following priority order:
    1. Explicitly provided `config_path` argument.
    2. The path specified by the 'CONFIG_PATH' environment variable.
    3. The default path: `<project_root>/config/config.yaml`.

    Args:
        config_path (str | None): Optional, explicit path to the configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration settings.

    Raises:
        FileNotFoundError: If the resolved configuration file path does not exist.
    """
    env_path = os.getenv("CONFIG_PATH")
    if config_path is None:
        config_path = env_path or str(_project_root() / "config" / "config.yaml")

    path = Path(config_path)
    if not path.is_absolute():
        path = _project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}