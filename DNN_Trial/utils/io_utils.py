import numpy as np
import os
import pickle
import datetime
import json
import yaml

output_dir = "output/"


def save_loss_to_file(args, arr, name, extension=""):
    filename = get_output_path(args, directory="logging", filename=name, extension=extension, file_type="txt")
    np.savetxt(filename, arr)
    #Check if file is created
    if os.path.exists(filename):
        print(f"Log file exists at: {filename}")
    else:
        print("Log file does NOT exist.")
        
    print(f"File name : {filename} . The file was saved")

def save_regularization_to_file(args, arr, name, extension=""):
    filename = get_output_path(args, directory="freq_reg", filename=name, extension=extension, file_type="txt")
    np.savetxt(filename, arr)


def save_predictions_to_file(arr, args, extension=""):
    filename = get_output_path(args, directory="predictions", filename="p", extension=extension, file_type="npy")
    np.save(filename, arr)


def save_model_to_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    pickle.dump(model, open(filename, 'wb'))


def load_model_from_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    return pickle.load(open(filename, 'rb'))


def save_results_to_json_file(args, jsondict, resultsname, append=True):
    """ Write the results to a json file. 
        jsondict: A dictionary with results that will be serialized.
        If append=True, the results will be appended to the original file.
        If not, they will be overwritten if the file already exists. 
    """
    filename = get_output_path(args, filename=resultsname, file_type="json")
    if append:
        if os.path.exists(filename):
            old_res = json.load(open(filename))
            for k, v in jsondict.items():
                old_res[k].append(v)
        else:
            old_res = {}
            for k, v in jsondict.items():
                old_res[k] = [v]
        jsondict = old_res
    json.dump(jsondict, open(filename, "w"))


def save_results_to_file(args, results, train_time=None, test_time=None, best_params=None):
    filename = get_output_path(args, filename="results", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write(args.model_name + " - " + args.dataset + "\n\n")

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        if train_time:
            text_file.write("\nTrain time: %f\n" % train_time)

        if test_time:
            text_file.write("Test time: %f\n" % test_time)

        if best_params:
            text_file.write("\nBest Parameters: %s\n\n\n" % best_params)


def save_hyperparameters_to_file(args, params, results, time=None):
    filename = get_output_path(args, filename="hp_log", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write("Parameters: %s\n\n" % params)

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        if time:
            text_file.write("\nTrain time: %f\n" % time[0])
            text_file.write("Test time: %f\n" % time[1])

        text_file.write("\n---------------------------------------\n")


def get_output_path(args, filename, file_type, directory=None, extension=None):
    # For example: output/LinearModel/Covertype
    dir_path = output_dir + args.model_name + "/" + args.dataset

    if directory:
        # For example: .../models
        dir_path = dir_path + "/" + directory

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + "/" + filename

    if extension is not None:
        file_path += "_" + str(extension)

    if file_type is not None:
        file_path += "." + file_type

    # For example: .../m_3.pkl

    return file_path


def get_predictions_from_file(args):
    dir_path = output_dir + args.model_name + "/" + args.dataset + "/predictions"

    files = os.listdir(dir_path)
    content = []

    for file in files:
        content.append(np.load(dir_path + "/" + file))

    return content



def update_yaml(dataset_name, model_name, parameters):
    file_path = "/home/mburu/Master_Thesis/master-thesis-da/DNN_Trial/config/results_params.yml"
    #file_path = '/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/DNN_Trial/config/results_params.yml'
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
