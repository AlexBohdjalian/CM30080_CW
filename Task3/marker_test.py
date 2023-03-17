import os
import time
import cv2
import traceback
from main import feature_detection_marker

BLUE = '\u001b[34m'
RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

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

def main_process_for_marker_test(test_dataset, training_dataset):
    try:
        # TODO: put best parameters here
        params = {
            'BFMatcher': {
                'crossCheck': False,
                'normType': 4
            },
            'RANSAC': {
                'confidence': 0.9423447867316301,
                'maxIters': 1500,
                'ransacReprojThreshold': 5.016592027133279
            },
            'ratioThreshold': 0.5011292212940329,
            'sift': {
                'contrastThreshold': 0.010041938693454615,
                'edgeThreshold': 15.355230618863885,
                'nOctaveLayers': 4,
                'nfeatures': 2000,
                'sigma': 1.934389681658647
            }
        }

        test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
        correct = 0
        false_pos = 0
        false_neg = 0
        for image, actual_feature_names_set in test_dataset:
            predicted_features = feature_detection_marker(image, all_training_data, params, show_output=True)
            predicted_feature_names_set = set([f[0] for f in predicted_features])

            # TODO: as per mark scheme, get True Positives, False Positives, etc.
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
        print('Accuracy:', accuracy)

    except Exception as e:
        print(RED, 'Unknown error occurred while processing images:', NORMAL, traceback.format_exc())
        exit()

# NOTE: For marker, we have assumed that the additional data you have is in the same format as the data given.
# Please replace the below three directories with your own and then execute this file.

no_rotation_images_dir = 'Task3/Task2Dataset/TestWithoutRotations/'
rotated_images_dir = 'Task3/Task3Dataset/'
training_images_dir = 'Task3/Task2Dataset/Training/'

try:
    all_no_rotation_images_and_features = read_test_dataset(no_rotation_images_dir, '.txt')
    all_rotation_images_and_features = read_test_dataset(rotated_images_dir, '.csv')
    all_training_data = read_training_dataset(training_images_dir)
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
    exit()

test_dataset = all_rotation_images_and_features + all_no_rotation_images_and_features
main_process_for_marker_test(test_dataset, all_training_data)
