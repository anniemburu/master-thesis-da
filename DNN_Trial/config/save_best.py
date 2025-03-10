import yaml

def update_yaml(file_path, dataset_name, model_name, parameters):
    # Load existing data
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file) or {}
    except FileNotFoundError:
        data = {}

    # Ensure structure exists
    if "parameters" not in data:
        data["parameters"] = {}
    if dataset_name not in data["parameters"]:
        data["parameters"][dataset_name] = {}
    if model_name not in data["parameters"][dataset_name]:
        data["parameters"][dataset_name][model_name] = {}

    # Overwrite the model parameters
    data["parameters"][dataset_name][model_name] = parameters  # This ensures overwriting

    # Save updated YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

# Example Usage
file_path = "/Users/wambo/Desktop/Master Thesis/master-thesis-da/DNN_Trial/config/results.yml"
dataset_name = "Boston"
model = "RandomForest"

# First update (initial parameters)
parameters_1 = {'max_depth': 12, 'n_estimators': 59}
update_yaml(file_path, dataset_name, model, parameters_1)

# Second update (overwrites previous parameters)
parameters_2 = {'max_depth': 20, 'n_estimators': 100, 'min_samples_split': 4}
update_yaml(file_path, dataset_name, model, parameters_2)
