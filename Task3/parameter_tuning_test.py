import os
import time
import cv2
import traceback
from main import feature_detection
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

BLUE = '\u001b[34m'
RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

task3_no_rotation_images_dir = 'Task3/Task2Dataset/TestWithoutRotations/'
task3_rotated_images_dir = 'Task3/Task3Dataset/'
task3_training_data_dir = 'Task3/Task2Dataset/Training/'

def read_no_rotations_dataset(dir):
    print(f'Reading dataset with no rotated images: {dir}')
    image_files = os.listdir(dir + 'images/')
    image_files = sorted(image_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

    all_data = []
    for image_file in image_files:
        csv_file = dir + 'annotations/' + image_file[:-4] + '.txt'
        with open(csv_file, 'r') as fr:
            features = fr.read().splitlines()
        all_features = []
        for feature in features:
            end_of_class_name_index = feature.find(", ")
            end_of_first_tuple_index = feature.find("), (") + 1
            feature_class = feature[:end_of_class_name_index]
            feature_coord1 = eval(feature[end_of_class_name_index + 2:end_of_first_tuple_index])
            feature_coord2 = eval(feature[end_of_first_tuple_index + 2:])

            all_features.append([feature_class, feature_coord1, feature_coord2])
        all_features.sort(key=lambda x:x[0])
        all_data.append([dir + 'images/' + image_file, all_features])

    return all_data

def read_rotations_dataset(dir):
    print(f'Reading dataset with rotated images: {dir}')
    image_files = os.listdir(dir + 'images/')
    image_files = sorted(image_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

    all_data = []
    for image_file in image_files:
        csv_file = dir + 'annotations/' + image_file[:-4] + '.csv'
        with open(csv_file, 'r') as fr:
            features = fr.read().splitlines()
        all_features = []
        for feature in features:
            end_of_class_name_index = feature.find(", ")
            end_of_first_tuple_index = feature.find("), (") + 1
            feature_class = feature[:end_of_class_name_index]
            feature_coord1 = eval(feature[end_of_class_name_index + 2:end_of_first_tuple_index])
            feature_coord2 = eval(feature[end_of_first_tuple_index + 2:])

            all_features.append([feature_class, feature_coord1, feature_coord2])
        all_features.sort(key=lambda x:x[0])
        all_data.append([dir + 'images/' + image_file, all_features])

    return all_data

def read_training_dataset(dir):
    print(f'Reading dataset with training images: {dir}')
    return [dir + 'png/' + path for path in os.listdir(dir + 'png/')]

try:
    all_no_rotation_images_and_features = read_no_rotations_dataset(task3_no_rotation_images_dir)
    all_rotation_images_and_features = read_rotations_dataset(task3_rotated_images_dir)
    all_training_data = read_training_dataset(task3_training_data_dir)
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
    exit()

def objective_mae(param):
    correct = 0
    # false_pos = 0
    # false_neg = 0
    for image_path, actual_features in test_dataset:
        predicted_features = sorted(feature_detection(image_path, all_training_data, param), key=lambda x:x[0])

        if len(predicted_features) == len(actual_features) and all(f1[0] == f2[0] for f1, f2 in zip(predicted_features, actual_features)):
            correct += 1
        # else:
        #     false_pos_res = [x[0] for x in predicted_features if x[0] not in [s[0] for s in actual_features]]
        #     false_neg_res = [x[0] for x in actual_features if x[0] not in [s[0] for s in predicted_features]]
        #     false_pos += any(false_pos_res)
        #     false_neg += any(false_neg_res)

    mae = (len(test_dataset) - correct) / len(test_dataset)

    return {'loss': mae, 'status': STATUS_OK, 'model': param}

try:
    # hp.choice, hp.uniform
    param_space = {
        'sift': {
            'nfeatures': hp.choice('nfeatures', [0, 1000, 2000]),
            'nOctaveLayers': hp.choice('nOctaveLayers', range(1, 6)),
            'contrastThreshold': hp.uniform('contrastThreshold', 0.01, 0.09),
            'edgeThreshold': hp.uniform('edgeThreshold', 5, 16),
            'sigma': hp.uniform('sigma', 0.8, 2.4),
        },
        'BFMatcher': {
            'normType': hp.choice('normType', [cv2.NORM_L2, cv2.NORM_L1]),
            'crossCheck': hp.choice('crossCheck', [True, False]),
        },
        'matchThreshold': hp.uniform('matchThreshold', 45, 55),
    }

    # test_dataset = all_no_rotation_images_and_features
    test_dataset = all_rotation_images_and_features
    # test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features

    trials = Trials()
    fmin(
        fn=objective_mae,
        space=param_space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials,
        timeout=None
    )
    
    best_params = trials.best_trial['result']['model']
except Exception as e:
    print(RED, 'Unknown error occurred: ', NORMAL, traceback.format_exc())
    exit()
