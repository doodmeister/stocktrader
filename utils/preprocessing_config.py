import json

def save_preprocessing_config(config: dict, path='preprocessing_config.json'):
    """Save preprocessing configuration to a JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f)

def load_preprocessing_config(path='preprocessing_config.json'):
    """Load preprocessing configuration from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)