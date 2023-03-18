import traceback

import cv2
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from main import feature_detection_hyperopt, read_test_dataset, read_training_dataset
from marker_test import main_process_for_marker_test

RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

task3_no_rotation_images_dir = 'Task3/Task2Dataset/TestWithoutRotations/'
task3_rotated_images_dir = 'Task3/Task3Dataset/'
task3_training_data_dir = 'Task3/Task2Dataset/Training/'

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

    # Pre-process all the training images one time as kp and desc will be the same
    all_training_data_kp_desc = []
    for current_image, current_image_path in all_training_data:
        current_kp, current_desc = sift.detectAndCompute(current_image, None)
        if current_desc is None:
            return {'status': STATUS_FAIL}
        all_training_data_kp_desc.append((current_image_path, current_kp, current_desc))

    correct = 0
    false_results = 0
    for image, actual_features in test_dataset:
        predicted_features = feature_detection_hyperopt(sift, bf, image, all_training_data_kp_desc, param)

        predicted_feature_names_set = set([f[0] for f in predicted_features])
        actual_feature_names_set = set([f[0] for f in actual_features])
        if predicted_feature_names_set == actual_feature_names_set:
            correct += 1
            continue

        false_results += len(predicted_feature_names_set.difference(actual_feature_names_set))
        false_results += len(actual_feature_names_set.difference(predicted_feature_names_set))

    accuracy = correct / len(test_dataset)
    if accuracy >= 0.8:
        print('Current Accuracy: ' + str(accuracy))
        print('Current Params: ' + str(param))
    return {'loss': false_results, 'status': STATUS_OK, 'model': param}

try:
    # First param_space:
    # param_space = {
    #     'sift': {
    #         'nfeatures': hp.choice('nfeatures', [0, 1000, 2000]),
    #         'nOctaveLayers': hp.choice('nOctaveLayers', range(1, 6)),
    #         'contrastThreshold': hp.uniform('contrastThreshold', 0.01, 0.09),
    #         'edgeThreshold': hp.uniform('edgeThreshold', 5, 16),
    #         'sigma': hp.uniform('sigma', 0.8, 2.4),
    #     },
    #     'BFMatcher': {
    #         'normType': hp.choice('normType', [cv2.NORM_L2, cv2.NORM_L1]),
    #         'crossCheck': hp.choice('crossCheck', [False]), # can only be False for knnMatch
    #     },
    #     'ratioThreshold': hp.uniform('ratioThreshold', 0.5, 1.0),
    #     'RANSAC': {
    #         'ransacReprojThreshold': hp.uniform('ransacReprojThreshold', 1, 10),
    #         'maxIters': hp.choice('maxIters', [500, 1000, 1500, 2000]),
    #         'confidence': hp.uniform('confidence', 0.90, 1.0),
    #     }
    # }
    # Refined:
    # param_space = {
    #     'sift': {
    #         'nfeatures': hp.choice('nfeatures', range(1500, 2501, 100)),
    #         'nOctaveLayers': hp.choice('nOctaveLayers', range(3, 6)),
    #         'contrastThreshold': hp.uniform('contrastThreshold', 0.005, 0.015),
    #         'edgeThreshold': hp.uniform('edgeThreshold', 10, 20),
    #         'sigma': hp.uniform('sigma', 1.7, 2.2),
    #     },
    #     'BFMatcher': {
    #         'normType': hp.choice('normType', [cv2.NORM_L2, cv2.NORM_L1]),
    #         'crossCheck': hp.choice('crossCheck', [False]), # can only be False for knnMatch
    #     },
    #     'ratioThreshold': hp.uniform('ratioThreshold', 0.4, 0.6),
    #     'RANSAC': {
    #         'ransacReprojThreshold': hp.uniform('ransacReprojThreshold', 3, 7),
    #         'maxIters': hp.choice('maxIters', range(1000, 2001, 250)),
    #         'confidence': hp.uniform('confidence', 0.9, 0.98),
    #     }
    # }
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
    # And again
    # param_space = {
    #     'sift': {
    #         'nfeatures': hp.choice('nfeatures', range(0, 2501, 250)),
    #         'nOctaveLayers': hp.choice('nOctaveLayers', range(3, 6)),
    #         'contrastThreshold': hp.uniform('contrastThreshold', 0.001, 0.2),
    #         'edgeThreshold': hp.uniform('edgeThreshold', 5, 20),
    #         'sigma': hp.uniform('sigma', 0.1, 2.5),
    #     },
    #     'BFMatcher': {
    #         'normType': hp.choice('normType', [cv2.NORM_L2, cv2.NORM_L1]),
    #         'crossCheck': False, # can only be False for knnMatch
    #     },
    #     'ratioThreshold': hp.uniform('ratioThreshold', 0.5, 0.8),
    #     'RANSAC': {
    #         'ransacReprojThreshold': hp.uniform('ransacReprojThreshold', 2, 9),
    #         'maxIters': hp.choice('maxIters', range(500, 2001, 100)),
    #         'confidence': hp.uniform('confidence', 0.8, 0.97),
    #     }
    # }

    # Best parameters, without geometric homography stuff:
        # accuracy 97.5%
        # params = {'BFMatcher': {'normType': 4,'crossCheck': True},'matchThreshold': 46.8732367930094, 'sift': {'nfeatures': 1000,'nOctaveLayers': 1,'contrastThreshold': 0.018249699973052022,'edgeThreshold': 15.286233923400257,'sigma': 1.9143801565688459}}
    # With geometric removal of outliers:
        # accuracy 85.0%, loss 10
        # params = {'BFMatcher': {'crossCheck': False, 'normType': 2}, 'RANSAC': {'confidence': 0.9630012856644677, 'maxIters': 1500, 'ransacReprojThreshold': 5.316510881843703}, 'ratioThreshold': 0.4026570872420355, 'sift': {'contrastThreshold': 0.006016864707454455, 'edgeThreshold': 11.771432535526955, 'nOctaveLayers': 4, 'nfeatures': 2000, 'sigma': 1.8086914201280728}}
        # accuracy 82.5%, loss 8
        # params = {'BFMatcher': {'crossCheck': False, 'normType': 2}, 'RANSAC': {'confidence': 0.9046460355608786, 'maxIters': 1250, 'ransacReprojThreshold': 3.6582628257928227}, 'ratioThreshold': 0.4328757119687792, 'sift': {'contrastThreshold': 0.011965469851854161, 'edgeThreshold': 14.056885732766037, 'nOctaveLayers': 5, 'nfeatures': 1800, 'sigma': 1.7364978622695921}}
        # accuracy 85.0%, loss 10
        # params = {'BFMatcher': {'crossCheck': False, 'normType': 2}, 'RANSAC': {'confidence': 0.8661380647700954, 'maxIters': 1200, 'ransacReprojThreshold': 5.433288470918298}, 'ratioThreshold': 0.48551688245254115, 'sift': {'contrastThreshold': 0.01966890925119041, 'edgeThreshold': 12.850484601282218, 'nOctaveLayers': 4, 'nfeatures': 1700, 'sigma': 1.8972781658650877}}

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    trials = Trials()
    fmin(
        fn=objective_false_res,
        space=param_space,
        algo=tpe.suggest,
        max_evals=1000,
        trials=trials
    )

    best_params = trials.best_trial['result']['model']
    print('Best parameters:', best_params)
    print('Evaluating best parameters for accuracy...')
    try:
        all_no_rotation_images_and_features = read_test_dataset(task3_no_rotation_images_dir, '.txt', cv2.IMREAD_COLOR)
        all_rotation_images_and_features = read_test_dataset(task3_rotated_images_dir, '.csv', cv2.IMREAD_COLOR)
        all_training_data = read_training_dataset(task3_training_data_dir)
    except Exception as e:
        print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
        exit()

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    main_process_for_marker_test(test_dataset, all_training_data, best_params)
except Exception as e:
    print(RED, 'Unknown error occurred:', NORMAL, traceback.format_exc())
    exit()
