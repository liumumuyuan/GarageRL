import yaml

def load_config(algo_name,file_path):
    with open(file_path,'r') as f:
        full_config = yaml.safe_load(f)

    shared = full_config.get('shared',{})
    algo_config = full_config.get(algo_name, {})
    merged_config = {**shared, **algo_config}
    return merged_config
