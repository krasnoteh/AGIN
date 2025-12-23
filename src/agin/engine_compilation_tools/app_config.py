import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AppConfig:
    path_to_models: Path
    engine_save_path: Path
    path_to_temp_onnx_models: Path
    delete_onnx_models_after_compilation: bool
    torch_dtype: torch.dtype
    height: int
    width: int
    resolution: Dict[str, int]
    raw_config: Dict[str, Any]  # Store original config for reference
    
    @classmethod
    def from_json(cls, config_path: str):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Parse torch_dtype
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64
        }
        
        torch_dtype_str = config_data.get('torch_dtype', 'float32')
        torch_dtype = dtype_mapping.get(torch_dtype_str, torch.float32)
        
        # Parse resolution
        resolution = config_data.get('resolution', {})
        height = resolution.get('height')
        width = resolution.get('width')
        
        # Parse paths and convert to Path objects
        path_to_models = Path(config_data['path_to_models'])
        engine_save_path = Path(config_data['engine_save_path'])
        path_to_temp_onnx_models = Path(config_data['path_to_temp_onnx_models'])
        
        # Parse boolean value
        delete_onnx_models = config_data.get('delete_onnx_models_after_compilation', True)
        
        return cls(
            path_to_models=path_to_models,
            engine_save_path=engine_save_path,
            path_to_temp_onnx_models=path_to_temp_onnx_models,
            delete_onnx_models_after_compilation=delete_onnx_models,
            torch_dtype=torch_dtype,
            height=height,
            width=width,
            resolution=resolution,
            raw_config=config_data
        )
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.path_to_models.mkdir(parents=True, exist_ok=True)
        self.engine_save_path.mkdir(parents=True, exist_ok=True)
        self.path_to_temp_onnx_models.mkdir(parents=True, exist_ok=True)