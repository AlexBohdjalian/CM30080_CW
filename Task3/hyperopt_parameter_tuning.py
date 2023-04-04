import traceback

import cv2
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from main import feature_detection_hyperopt, main_process_for_marker, read_test_dataset, read_training_dataset


RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

task3_no_rotation_images_dir = 'Task3/Task2Dataset/TestWithoutRotations/'
task3_rotated_images_dir = 'Task3/Task3Dataset/'
task3_training_data_dir = 'Task3/Task2Dataset/Training/'

try:
    all_no_rotation_images_and_features = read_test_dataset(task3_no_rotation_images_dir, '.txt')
    all_rotation_images_and_features = read_test_dataset(task3_rotated_images_dir, '.csv')
    all_training_images_and_paths = read_training_dataset(task3_training_data_dir)
    print()
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
    exit()

def objective_false_res(param):
    sift = cv2.SIFT_create(**param['sift'])
    bf = cv2.BFMatcher(**param['BFMatcher'])

    # Pre-process all the training images one time as kp and desc will be the same
    all_training_data_kp_desc = []
    for current_image, current_image_path in all_training_images_and_paths:
        current_kp, current_desc = sift.detectAndCompute(current_image, None)
        if current_desc is None:
            return {'status': STATUS_FAIL}
        all_training_data_kp_desc.append((current_image_path, current_kp, current_desc))

    correct = 0
    false_results_count = 0
    for gray_image, actual_features in test_dataset:
        predicted_features = feature_detection_hyperopt(sift, bf, gray_image, all_training_data_kp_desc, param)

        predicted_feature_names_set = set([f[0] for f in predicted_features])
        actual_feature_names_set = set([f[0] for f in actual_features])
        if predicted_feature_names_set == actual_feature_names_set:
            correct += 1
            continue

        false_results_count += len(predicted_feature_names_set.difference(actual_feature_names_set))
        false_results_count += len(actual_feature_names_set.difference(predicted_feature_names_set))

    accuracy = correct / len(test_dataset)
    if accuracy > 0.85 or false_results_count < 9:
        print(f'Current Accuracy: {accuracy}, loss: {false_results_count}')
        print('Current Params: ' + str(param))
    return {'loss': false_results_count, 'status': STATUS_OK, 'model': param}

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
    # Refined based on best results from above param_space:
    param_space = {
        'BFMatcher': {
            # TODO: experiment with using knnmatch and match. use if statement to determine using this param
            'crossCheck': False,
            'normType': 2
        },
        'RANSAC': {
            'confidence': hp.uniform('confidence', 0.955, 0.97), # 0.9623804736073822 = mean of [0.9611536048413046, 0.962462452378438, 0.962524362602404]
            'maxIters': hp.choice('maxIters', range(1400, 1601, 50)), # 1500 = mode of [1600, 1400, 1500]
            'ransacReprojThreshold': hp.uniform('ransacReprojThreshold', 5.4, 5.6), # 5.488956712586637 = mean of [5.401179963009146, 5.5846992491264364, 5.478990924625329]
        },
        'min_good_matches': 4,
        'ratioThreshold': hp.uniform('ratioThreshold', 0.41, 0.4265),# 0.41876768654455434 = mean of [0.4166608892456559, 0.4191789633890037, 0.42046320699900236]
        'sift': {
            'nfeatures': 2100,
            'nOctaveLayers': 4,
            'contrastThreshold': hp.uniform('contrastThreshold', 0.005, 0.006), # 0.00553 = mean of [0.005509519406815533, 0.005544726718043102, 0.0055376019160449244]
            'edgeThreshold': hp.uniform('edgeThreshold', 11, 13), # 12.332041427222738 = mean of [11.848464359070503, 12.059998508426833, 12.087660416130879]
            'sigma': hp.uniform('sigma', 1.7, 1.9), # 1.808201349714444 = mean of [1.8028239914595838, 1.811168377720828, 1.812710677962918]
        }
    }

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    trials = Trials()
    fmin(
        fn=objective_false_res,
        space=param_space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials
    )

    best_params = trials.best_trial['result']['model']
    print('Best parameters:', best_params)
    print('Evaluating best parameters for accuracy...')
    try:
        # Re-read these in colour
        all_no_rotation_images_and_features = read_test_dataset(task3_no_rotation_images_dir, '.txt', read_colour=True)
        all_rotation_images_and_features = read_test_dataset(task3_rotated_images_dir, '.csv', read_colour=True)
    except Exception as e:
        print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
        exit()

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    main_process_for_marker(test_dataset, all_training_images_and_paths, best_params)
except Exception as e:
    print(RED, 'Unknown error occurred:', NORMAL, traceback.format_exc())
    exit()
