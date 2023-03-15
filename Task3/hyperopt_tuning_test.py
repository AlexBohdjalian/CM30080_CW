import os
import cv2
import traceback
from main import feature_detection_hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

RED = '\u001b[31m'
NORMAL = '\u001b[0m'

task3_no_rotation_images_dir = 'Task3/Task2Dataset/TestWithoutRotations/'
task3_rotated_images_dir = 'Task3/Task3Dataset/'
task3_training_data_dir = 'Task3/Task2Dataset/Training/'

def read_test_dataset(dir, file_ext):
    print(f'Reading test dataset: {dir}')
    image_files = os.listdir(dir + 'images/')
    image_files = sorted(image_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

    all_data = []
    for image_file in image_files:
        csv_file = dir + 'annotations/' + image_file[:-4] + file_ext
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
        path = dir + 'images/' + image_file
        all_data.append([cv2.imread(path, cv2.IMREAD_GRAYSCALE), set([f[0] for f in all_features])])

    return all_data

def read_training_dataset(dir):
    print(f'Reading training dataset: {dir}')
    return [(cv2.imread(dir + 'png/' + path, cv2.IMREAD_GRAYSCALE), path) for path in os.listdir(dir + 'png/')]

try:
    all_no_rotation_images_and_features = read_test_dataset(task3_no_rotation_images_dir, '.txt')
    all_rotation_images_and_features = read_test_dataset(task3_rotated_images_dir, '.csv')
    all_training_data = read_training_dataset(task3_training_data_dir)
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
    exit()

def objective_mae(param):
    sift = cv2.SIFT_create(
        nfeatures=param['sift']['nfeatures'],
        nOctaveLayers=param['sift']['nOctaveLayers'],
        contrastThreshold=param['sift']['contrastThreshold'],
        edgeThreshold=param['sift']['edgeThreshold'],
        sigma=param['sift']['sigma']
    )
    bf = cv2.BFMatcher(
        normType=param['BFMatcher']['normType'],
        crossCheck=param['BFMatcher']['crossCheck']
    )

    all_training_data_kp_desc = []
    for current_image, current_image_path in all_training_data:
        current_kp, current_desc = sift.detectAndCompute(current_image, None)
        all_training_data_kp_desc.append((current_image_path, current_kp, current_desc))

    wrong = 0
    for image, actual_feature_names_set in test_dataset:
        predicted_features = feature_detection_hyperopt(
            sift,
            bf,
            image,
            all_training_data_kp_desc,
            param['matchThreshold'],
            doBoundingBox=True
        )

        predicted_feature_names_set = set([f[0] for f in predicted_features])
        if predicted_feature_names_set != actual_feature_names_set:
            wrong += 1

    mae = wrong / len(test_dataset)

    return {'loss': mae, 'status': STATUS_OK, 'model': param}

try:
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
    # Best parameters: 97.5% accuracy
    # param_space = {
    #     'sift': {
    #         'nfeatures': 1000,
    #         'nOctaveLayers': 1,
    #         'contrastThreshold': 0.018249699973052022,
    #         'edgeThreshold': 15.286233923400257,
    #         'sigma': 1.9143801565688459,
    #     },
    #     'BFMatcher': {
    #         'normType': 4,
    #         'crossCheck': True,
    #     },
    #     'matchThreshold': 46.8732367930094,
    # }

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    # TODO: shuffle these?

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
    print(best_params)
except Exception as e:
    print(RED, 'Unknown error occurred: ', NORMAL, traceback.format_exc())
    exit()
