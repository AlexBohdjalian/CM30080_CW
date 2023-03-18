import traceback

import cv2
import numpy as np
from main import feature_detection_marker, read_test_dataset, read_training_dataset

BLUE = '\u001b[34m'
RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

def main_process_for_marker_test(marker_test_dataset, training_dataset, params):
    try:
        sift = cv2.SIFT_create(**params['sift'])
        bf = cv2.BFMatcher(**params['BFMatcher'])

        correct = 0
        false_pos = 0
        false_neg = 0
        times = []
        for i in range(len(test_dataset)):
            colour_query_image, actual_features = marker_test_dataset[i]

            print(f'Test Image {i} ...')
            predicted_features, run_time = feature_detection_marker(
                sift,
                bf,
                colour_query_image,
                training_dataset,
                params,
                show_output=False # TODO: set to True before submitting
            )

            times.append(run_time)
            predicted_feature_names_set = set([f[0] for f in predicted_features])
            actual_feature_names_set = set([f[0] for f in actual_features])

            # TODO: as per mark scheme, get True Positives, False Positives, etc.
            if actual_feature_names_set == predicted_feature_names_set:
                correct += 1
                print(GREEN, 'Correct!!!', NORMAL)
            elif predicted_feature_names_set != actual_feature_names_set:
                false_pos += len(predicted_feature_names_set.difference(actual_feature_names_set))
                false_neg += len(actual_feature_names_set.difference(predicted_feature_names_set))
                print(RED, 'IN-Correct!!!', NORMAL)

            print('Predicted:', predicted_feature_names_set)
            print('Actual   :', actual_feature_names_set)
            print()

        accuracy = correct * 100 / len(list(test_dataset))
        total_false_results = false_pos + false_neg
        print(f'Accuracy: {accuracy}%')
        print(f'Total false results: {total_false_results}')
        print(f'Average runtime per image: {round(np.mean(times), 3)}ms')
        print(BLUE, f'Note: The average runtime is for the feature matching process '
              + 'and does not include any additional processing done to check '
              + 'accuracy, display the results, etc.', NORMAL)
    except:
        print(RED, 'Unknown error occurred:', NORMAL, traceback.format_exc())
        exit()

if __name__ == '__main__':
    params = {
        'BFMatcher': {
            'crossCheck': False,
            'normType': 2
        },
        'RANSAC': {
            'confidence': 0.9630012856644677,
            'maxIters': 1500,
            'ransacReprojThreshold': 5.316510881843703
        },
        'ratioThreshold': 0.4026570872420355,
        'sift': {
            'contrastThreshold': 0.006016864707454455,
            'edgeThreshold': 11.771432535526955,
            'nOctaveLayers': 4,
            'nfeatures': 2000,
            'sigma': 1.8086914201280728
        }
    }

    # NOTE: For marker, we have assumed that the additional data you have is in the same format as the data given.
    # Please replace the below three directories with your own and then execute this file.
    no_rotation_images_dir = 'Task3/Task2Dataset/TestWithoutRotations/'
    rotated_images_dir = 'Task3/Task3Dataset/'
    training_images_dir = 'Task3/Task2Dataset/Training/'

    try:
        all_no_rotation_images_and_features = read_test_dataset(no_rotation_images_dir, '.txt', read_colour=True)
        all_rotation_images_and_features = read_test_dataset(rotated_images_dir, '.csv', read_colour=True)
        all_training_data = read_training_dataset(training_images_dir)
    except Exception as e:
        print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
        exit()

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    main_process_for_marker_test(test_dataset, all_training_data, params)
