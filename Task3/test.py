import os
from main import feature_detection
import time


BLUE = '\u001b[34m'
RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

task3_task2Dataset_dir = 'Task3/Task2Dataset/'
task3_task3Dataset_dir = 'Task3/Task3Dataset/'

def read_no_rotations_dataset(dir):
    print(f'Reading dataset {dir} ...')
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

def read_training_dataset(dir):
    print(f'Reading dataset {dir} ...')
    return [dir + 'png/' + path for path in os.listdir(dir + 'png/')]

def read_rotations_dataset(dir):
    print(f'Reading dataset {dir} ...')
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

try:
    all_no_rotation_images_and_features = read_no_rotations_dataset(task3_task2Dataset_dir + 'TestWithoutRotations/')
    all_training_data = read_training_dataset(task3_task2Dataset_dir + 'Training/')
    all_rotation_images_and_features = read_rotations_dataset(task3_task3Dataset_dir)
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, e)
    exit()

print('Processing images ...')
try:
    params_list = [{'nfeatures': nf, 'nOctaveLayers': nl, 'contrastThreshold': ct, 'edgeThreshold': et, 'sigma': s}
        for nf in [0]
        for nl in [2, 3, 4, 5]
        for ct in [0.01, 0.03, 0.05, 0.07, 0.09]
        for et in [5, 10, 15, 20]
        for s in [1.2, 1.6, 2.0]
    ]

    best_acc = 0
    best_params = {}
    start_t = time.time()
    test_dataset = all_no_rotation_images_and_features# + all_rotation_images_and_features
    for i in range(len(params_list)):
        print(f'Grid-search progress: {i + 1}/{len(params_list)}\t ...', end='\r')
        correct = []
        wrong = []
        for image_path, actual_features in test_dataset: # replace this with suitable dataset (or combined)
            predicted_features = sorted(feature_detection(image_path, all_training_data, params_list[i]), key = lambda x : x[0])
            # print('For:', image_path)
            # print('Predicted:', [feature[0] for feature in predicted_features])
            # print('Actual   :', [feature[0] for feature in actual_features])

            # TODO: split this out to recognise, false positives, false negatives
            if len(predicted_features) == len(actual_features) and \
                all([f1[0] == f2[0] for f1, f2 in zip(predicted_features, actual_features)]):
                    correct.append(image_path)
                    # print(GREEN, 'Correct!!!', NORMAL)
            else:
                # false_pos = [x[0] for x in predicted_features if x[0] not in [s[0] for s in actual_features]]
                # false_neg = [x[0] for x in actual_features if x[0] not in [s[0] for s in predicted_features]]
                wrong.append(image_path)
                # print(RED, 'False pos:', false_pos, NORMAL)
                # print(RED, 'False neg:', false_neg, NORMAL)
                # print(RED, 'IN-Correct!!!', NORMAL)

            # except Exception as e:
            #     print(RED, 'Unknown error occurred while testing image:', image_path, NORMAL, e)
            #     errors.append(image_path)
            #     exit()

        accuracy = len(correct) * 100 / len(list(test_dataset))
        if accuracy > best_acc:
            best_acc = accuracy
            best_params = params_list[i]

        print(f'Grid-search progress: {i + 1}/{len(params_list)}\t Accuracy: {round(accuracy, 1)}%\tTime: {round(time.time() - start_t)}s')

        # print(BLUE)
        # print(f'Correct: {len(correct)}')
        # print(f'Wrong: {len(wrong)}')
        # print(f'Errors: {len(errors)}')
        # print(f'Accuracy: {round(accuracy, 1)}%')
        # print(NORMAL)

    print(f'\nBest Params: {best_params}')
    print(f'\nBest Params acc: {best_acc}')
except Exception as e:
    print(RED, 'Unknown error occurred while processing images:', NORMAL, e)
    exit()
