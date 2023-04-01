import traceback

from main import feature_detection, read_test_dataset, read_training_dataset

RED = '\u001b[31m'
NORMAL = '\u001b[0m'


if __name__ == '__main__':
    params = {'RANSAC': {'ransacReprojThreshold': 11.110294305510669},
              'SIFT': {'contrastThreshold': 0.0039052330228148877,
                       'edgeThreshold': 16.379139206562137,
                       'nOctaveLayers': 6,
                       'nfeatures': 1700,
                       'sigma': 2.2201211013686857},
              'BF': {'crossCheck': False,
                     'normType': 2},
              'inlierScore': 4,
              'ratioThreshold': 0.6514343913409797,
              'resizeQuery': 95}

    # NOTE: For marker, we have assumed that the additional data you have is in the same format as the data given.
    # Please replace the below three directories with your own and then execute this file.
    train_data_dir = 'Task2Dataset/Training/'
    query_data_dir = 'Task3Dataset/'

    try:
        train_data = read_training_dataset(train_data_dir)
        query_data = read_test_dataset(query_data_dir, '.csv')
    except Exception as e:
        print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
        exit()

    # set the show_output to True to see the query images with the detected features
    # if set to true ignore 'Avg. time per query image' metric
    feature_detection(train_data, query_data, params, show_output=False)

"""

if __name__ == '__main__':
    params = {
        'BFMatcher': {
            'crossCheck': False,
            'normType': 2
        },
        'RANSAC': {
            'confidence': 0.9564838900729838,
            'maxIters': 1600,
            'ransacReprojThreshold': 5.53440270211734
        },
        'min_good_matches': 4,
        'ratioThreshold': 0.42352058295191136,
        'sift': {
            'contrastThreshold': 0.005457729696636313,
            'edgeThreshold': 11.188051836654086,
            'nOctaveLayers': 4,
            'nfeatures': 2100,
            'sigma': 1.8708988402771627
        }
    }
    # Tested to have an accuracy of 85% with a total of 0 false-positives and 9 false negatives all 40 images.

    # NOTE: For marker, we have assumed that the additional data you have is in the same format as the data given.
    # Please replace the below three directories with your own and then execute this file.
    no_rotation_images_dir = 'Task3/Task2Dataset/TestWithoutRotations/'
    rotated_images_dir = 'Task3/Task3Dataset/'
    training_images_dir = 'Task3/Task2Dataset/Training/'

    try:
        all_no_rotation_images_and_features = read_test_dataset(no_rotation_images_dir, '.txt', read_colour=True)
        all_rotation_images_and_features = read_test_dataset(rotated_images_dir, '.csv', read_colour=True)
        all_training_images_and_paths = read_training_dataset(training_images_dir)
        print()
    except Exception as e:
        print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
        exit()

    test_images_and_features = all_no_rotation_images_and_features + all_rotation_images_and_features
    main_process_for_marker(
        test_images_and_features,
        all_training_images_and_paths,
        params,
        show_output=False # TODO: set this to true before submitting
    )

"""


