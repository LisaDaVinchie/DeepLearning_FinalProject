import inspect
import typing

def typecast_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise ValueError(f"Invalid value for boolean: {value}")


# Filter parameters to match the model's __init__ arguments
def filter_params(model_class, json_params, params_to_skip=["self", "in_channels"]):
    # Get the parameter names of the class's __init__ method
    sig = inspect.signature(model_class.__init__)
    model_params = sig.parameters.items()
    
    # Assuming that the class and json params have the same keys
    valid_params = {}
    for model_key, model_param in model_params:
        # Skip self and in_channels
        if model_param.name in params_to_skip:
            continue
        expected_type = model_param.annotation
        
        if model_key in json_params:
            value = json_params[model_key]
            expected_type = model_param.annotation

            # Handle type conversion for known types (e.g., list, int)
            if expected_type != inspect.Parameter.empty:
                try:
                    if isinstance(value, str):
                        # Check if the expected type is a list or tuple
                        if typing.get_origin(expected_type) in [list, tuple]:
                            # Split string into a list
                            args_type = typing.get_args(expected_type)
                            # Convert elements to the correct type (int, float, etc.)
                            value: typing.List = [args_type[0](v) for v in value.split(" ")]
                        elif expected_type == bool:
                            value = typecast_bool(value)
                        else:
                            value = expected_type(value)  # Convert single value to the expected type
                    valid_params[model_key] = value
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Failed to convert parameter {model_key} with value {value} to type {expected_type}. Error: {e}")
            else:
                valid_params[model_key] = value  # If 
        else:
            print(f"Parameter {model_key} not found in the input parameters")

            
    return valid_params