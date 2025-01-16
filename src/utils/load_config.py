from pathlib import Path
import json

def _typecast_value(value, key, type_info):
    """Typecast the value based on the type_info loaded from JSON."""
    cast_type = type_info.get(key, "str")  # Default to string if not found in type_info
    if cast_type == "int":
        return int(value)
    elif cast_type == "float":
        return float(value)
    elif cast_type == "bool":
        return value.lower() == "true"
    return value

def _process_params(params, type_info):
    processed_params = {}
    for key, value in params.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            # If the value is a nested dictionary, recurse
            processed_params[key] = _process_params(value, type_info)
        else:
            # Otherwise, typecast the value
            processed_params[key] = _typecast_value(value, key, type_info)
    return processed_params

def load_params(config_path: Path, params_subset: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    type_info = config["type_info"]
    sub_params = config[params_subset]
    
    params = _process_params(config[params_subset], type_info)
    
    return params