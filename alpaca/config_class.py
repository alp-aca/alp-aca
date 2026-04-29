import tomllib
from pathlib import Path
import os
import sys

if sys.platform == "win32":
    _base = Path(os.environ.get("APPDATA", "~")).expanduser()
else:
    _base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
_config_file = _base / "alpaca" / "config.toml"

class AlpacaConfig:
    """Configuration class for ALP-aca."""

    def __init__(self,
        vegas_seed: int = 12345,
    ):
        self.vegas_seed = vegas_seed
        


    @property
    def vegas_seed(self) -> int:
        """Seed for the VEGAS algorithm."""
        return self._vegas_seed

    @vegas_seed.setter
    def vegas_seed(self, value: int):
        import gvar
        gvar.ranseed(value)
        self._vegas_seed = value

    def save(self):
        """Save the configuration to a TOML file."""
        config_data = {
            "vegas_seed": self.vegas_seed,
        }

        lines = [f"{key} = {value!r}" for key, value in config_data.items()]
        _config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_config_file, "w") as f:
            f.write("\n".join(lines))

    @classmethod
    def load(cls) -> "AlpacaConfig":
        """Load the configuration from a TOML file."""
        if not _config_file.exists():
            return cls()
        with open(_config_file, "rb") as f:
            config_data = tomllib.load(f)
        return cls(**config_data)

config = AlpacaConfig.load()