import os
import time
import cv2
import traceback
from main import feature_detection

BLUE = '\u001b[34m'
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
    return [(cv2.imread(dir + 'png/' + path, cv2.IMREAD_GRAYSCALE), path) for path in os.listdir(dir + 'png/')]

try:
    all_no_rotation_images_and_features = read_test_dataset(task3_no_rotation_images_dir, '.txt')
    all_rotation_images_and_features = read_test_dataset(task3_rotated_images_dir, '.csv')
    all_training_data = read_training_dataset(task3_training_data_dir)
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
    exit()

try:
    # TODO: put best parameters here
    # TODO: re-write this file for markers to use on their own test data
    param_space = {
        'sift': {
            'nfeatures': 1000,
            'nOctaveLayers': 1,
            'contrastThreshold': 0.018249699973052022,
            'edgeThreshold': 15.286233923400257,
            'sigma': 1.9143801565688459,
        },
        'BFMatcher': {
            'normType': 4,
            'crossCheck': True,
        },
        'matchThreshold': 46.8732367930094,
    }

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    print(f'There are {len(test_dataset)} images to classify ...')
    correct = 0
    false_pos = 0
    false_neg = 0
    false_pos_neg = 0
    for image, actual_feature_names_set in test_dataset: # replace this with suitable dataset (or combined)
        predicted_features = feature_detection(image, all_training_data, param_space, show_output=True)
        predicted_feature_names_set = set([f[0] for f in predicted_features])

        if actual_feature_names_set == predicted_feature_names_set:
            correct += 1
            print(GREEN, 'Correct!!!', NORMAL)
        elif predicted_feature_names_set != actual_feature_names_set:
            false_pos_dif = predicted_feature_names_set.difference(actual_feature_names_set)
            false_neg_dif = actual_feature_names_set.difference(predicted_feature_names_set)
            if any(false_pos_dif) and any(false_neg_dif):
                false_pos_neg += 1
            elif any(false_pos_dif):
                false_pos += 1
            elif any(false_neg_dif):
                false_neg += 1
            print(RED, 'IN-Correct!!!', NORMAL)

        print('Predicted:', predicted_feature_names_set)
        print('Actual   :', actual_feature_names_set)

    accuracy = correct * 100 / len(list(test_dataset))

except Exception as e:
    print(RED, 'Unknown error occurred while processing images:', NORMAL, traceback.format_exc())
    exit()

# # Tidy-up
# if os.path.exists(task3_training_data_dir + tmp_file_dir):
#     shutil.rmtree(task3_training_data_dir + tmp_file_dir)
