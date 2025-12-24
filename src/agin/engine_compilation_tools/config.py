import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    path_to_models: Path
    engine_save_path: Path
    height: int
    width: int
    resolution: Dict[str, int]
    raw_config: Dict[str, Any]
    
    @classmethod
    def from_json(cls, config_path: str):
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Parse resolution
        resolution = config_data.get('resolution', {})
        height = resolution.get('height')
        width = resolution.get('width')
        
        # Parse paths and convert to Path objects
        path_to_models = Path(config_data['path_to_models'])
        engine_save_path = Path(config_data['engine_save_path'])
        
        return cls(
            path_to_models=path_to_models,
            engine_save_path=engine_save_path,
            height=height,
            width=width,
            resolution=resolution,
            raw_config=config_data
        )
    
    def __post_init__(self):
        self.engine_save_path.mkdir(parents=True, exist_ok=True)
