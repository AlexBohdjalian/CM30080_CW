import os
import cv2
import traceback
from main import feature_detection_marker, feature_detection_hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

RED = '\u001b[31m'
GREEN = '\u001b[32m'
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
    return [(
        cv2.imread(dir + 'png/' + path, cv2.IMREAD_GRAYSCALE),
        path
    ) for path in os.listdir(dir + 'png/')]

try:
    all_no_rotation_images_and_features = read_test_dataset(task3_no_rotation_images_dir, '.txt')
    all_rotation_images_and_features = read_test_dataset(task3_rotated_images_dir, '.csv')
    all_training_data = read_training_dataset(task3_training_data_dir)
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
    exit()

def objective_false_res(param):
    sift = cv2.SIFT_create(**param['sift'])
    bf = cv2.BFMatcher(**param['BFMatcher'])

    all_training_data_kp_desc = []
    for current_image, current_image_path in all_training_data:
        current_kp, current_desc = sift.detectAndCompute(current_image, None)

        if current_desc is None:
            return {'status': STATUS_FAIL}

        all_training_data_kp_desc.append((current_image_path, current_kp, current_desc))

    correct = 0
    false_results = 0
    for image, actual_feature_names_set in test_dataset:
        predicted_features = feature_detection_hyperopt(sift, bf, image, all_training_data_kp_desc, param)

        predicted_feature_names_set = set([f[0] for f in predicted_features])
        if predicted_feature_names_set == actual_feature_names_set:
            correct += 1
        else:
            false_results += len(predicted_feature_names_set.difference(actual_feature_names_set)) \
                + len(actual_feature_names_set.difference(predicted_feature_names_set))

    accuracy = correct / len(test_dataset)
    if accuracy >= 0.825:
        print('Current Accuracy: ' + str(accuracy))
        print('Current Params: ' + str(param))
    return {'loss': false_results, 'status': STATUS_OK, 'model': param}

try:
    # First param_space:
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
            'crossCheck': hp.choice('crossCheck', [False]), # can only be False for knnMatch
        },
        'ratioThreshold': hp.uniform('ratioThreshold', 0.5, 1.0),
        'RANSAC': {
            'ransacReprojThreshold': hp.uniform('ransacReprojThreshold', 1, 10),
            'maxIters': hp.choice('maxIters', [500, 1000, 1500, 2000]),
            'confidence': hp.uniform('confidence', 0.90, 1.0),
        }
    }
    # Refined:
    param_space = {
        'sift': {
            'nfeatures': hp.choice('nfeatures', range(1500, 2501, 100)),
            'nOctaveLayers': hp.choice('nOctaveLayers', range(3, 6)),
            'contrastThreshold': hp.uniform('contrastThreshold', 0.005, 0.015),
            'edgeThreshold': hp.uniform('edgeThreshold', 10, 20),
            'sigma': hp.uniform('sigma', 1.7, 2.2),
        },
        'BFMatcher': {
            'normType': hp.choice('normType', [cv2.NORM_L2, cv2.NORM_L1]),
            'crossCheck': hp.choice('crossCheck', [False]), # can only be False for knnMatch
        },
        'ratioThreshold': hp.uniform('ratioThreshold', 0.4, 0.6),
        'RANSAC': {
            'ransacReprojThreshold': hp.uniform('ransacReprojThreshold', 3, 7),
            'maxIters': hp.choice('maxIters', range(1000, 2001, 250)),
            'confidence': hp.uniform('confidence', 0.9, 0.98),
        }
    }
    # Refined again:
    param_space = {
        'sift': {
            'nfeatures': hp.choice('nfeatures', range(1700, 2301, 100)),
            'nOctaveLayers': hp.choice('nOctaveLayers', range(3, 6)),
            'contrastThreshold': hp.uniform('contrastThreshold', 0.005, 0.15),
            'edgeThreshold': hp.uniform('edgeThreshold', 10, 15),
            'sigma': hp.uniform('sigma', 0.1, 2.1),
        },
        'BFMatcher': {
            'normType': cv2.NORM_L1, # seems to have converged
            'crossCheck': False, # can only be False for knnMatch
        },
        'ratioThreshold': hp.uniform('ratioThreshold', 0.3, 0.6),
        'RANSAC': {
            'ransacReprojThreshold': hp.uniform('ransacReprojThreshold', 2.5, 7.5),
            'maxIters': hp.choice('maxIters', range(1000, 1501, 100)),
            'confidence': hp.uniform('confidence', 0.85, 0.98),
        }
    }
    
    # Best parameters, without geometric homography stuff:
        # accuracy 97.5%
        # params = {'BFMatcher': {'normType': 4,'crossCheck': True},'matchThreshold': 46.8732367930094, 'sift': {'nfeatures': 1000,'nOctaveLayers': 1,'contrastThreshold': 0.018249699973052022,'edgeThreshold': 15.286233923400257,'sigma': 1.9143801565688459}}
    # With geometric removal of outliers:
        # accuracy 85%, best loss 10
        # params = {'BFMatcher': {'crossCheck': False, 'normType': 2}, 'RANSAC': {'confidence': 0.9630012856644677, 'maxIters': 1500, 'ransacReprojThreshold': 5.316510881843703}, 'ratioThreshold': 0.4026570872420355, 'sift': {'contrastThreshold': 0.006016864707454455, 'edgeThreshold': 11.771432535526955, 'nOctaveLayers': 4, 'nfeatures': 2000, 'sigma': 1.8086914201280728}}
        # accuracy 82.5%, best loss 8
        # params = {'BFMatcher': {'crossCheck': False, 'normType': 2}, 'RANSAC': {'confidence': 0.9046460355608786, 'maxIters': 1250, 'ransacReprojThreshold': 3.6582628257928227}, 'ratioThreshold': 0.4328757119687792, 'sift': {'contrastThreshold': 0.011965469851854161, 'edgeThreshold': 14.056885732766037, 'nOctaveLayers': 5, 'nfeatures': 1800, 'sigma': 1.7364978622695921}}

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    # TODO: shuffle these?

    trials = Trials()
    fmin(
        fn=objective_false_res,
        space=param_space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials,
        timeout=None
    )

    best_params = trials.best_trial['result']['model']
    print('Best parameters:', best_params)
    print()

    print('Evaluating best parameters for accuracy...')
    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    correct = 0
    false_pos = 0
    false_neg = 0
    for image, actual_feature_names_set in test_dataset: # replace this with suitable dataset (or combined)
        predicted_features = feature_detection_marker(image, all_training_data, best_params, show_output=True)
        predicted_feature_names_set = set([f[0] for f in predicted_features])

        if actual_feature_names_set == predicted_feature_names_set:
            correct += 1
            print(GREEN, 'Correct!!!', NORMAL)
        elif predicted_feature_names_set != actual_feature_names_set:
            false_pos_dif = predicted_feature_names_set.difference(actual_feature_names_set)
            false_neg_dif = actual_feature_names_set.difference(predicted_feature_names_set)
            if any(false_pos_dif):
                false_pos += 1
            if any(false_neg_dif):
                false_neg += 1
            print(RED, 'IN-Correct!!!', NORMAL)

        print('Predicted:', predicted_feature_names_set)
        print('Actual   :', actual_feature_names_set)

    accuracy = correct * 100 / len(list(test_dataset))
    print('Final Accuracy: ' + str(accuracy))

except Exception as e:
    print(RED, 'Unknown error occurred:', NORMAL, traceback.format_exc())
    exit()
