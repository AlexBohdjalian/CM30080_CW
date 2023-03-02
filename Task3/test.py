import os
from main import training_process, feature_detection

BLUE = '\u001b[34m'
RED = '\u001b[31m'
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
            end_of_class = feature.find(", ")
            end_of_tuple = feature.find("), (") + 1
            feature_class = feature[:end_of_class]
            feature_coord1 = eval(feature[end_of_class + 2:end_of_tuple])
            feature_coord2 = eval(feature[end_of_tuple + 2:])

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
            end_of_class = feature.find(", ")
            end_of_tuple = feature.find("), (") + 1
            feature_class = feature[:end_of_class]
            feature_coord1 = eval(feature[end_of_class + 2:end_of_tuple])
            feature_coord2 = eval(feature[end_of_tuple + 2:])

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
    print(NORMAL)
    exit()

print('Processing images...')
try:
    correct = []
    wrong = []
    errors = []
    test_dataset = all_no_rotation_images_and_features
    for image_path, actual_features in test_dataset: # replace this with suitable dataset (or combined)
        try:
            predicted_features = sorted(feature_detection(image_path, all_training_data), key = lambda x : x[0])
            print('For:', image_path)
            print('Predicted:', predicted_features)
            print('Actual   :', actual_features)
            print()
            # print('Match between query descriptor {} and target descriptor {}: distance = {}'.format(
            #     predicted_features.queryIdx,
            #     predicted_features.trainIdx,
            #     predicted_features.distance
            # ))

            # TODO: change this to match class names and loosely match bounding boxes?
            actual_feature_names = [feature[0] for feature in actual_features]
            is_correct = True
            for feature in predicted_features:
                if feature[0] not in actual_feature_names:
                    is_correct = False
                    break

            if is_correct:
                correct.append(image_path)
                # print('Correct!!!')
            else:
                wrong.append(image_path)
                # print('IN-Correct!!!')

        except Exception as e:
            print(RED, 'Unknown error occurred while testing image:', image_path, NORMAL, e)
            errors.append(image_path)

    BLUE = '\u001b[34m'
    NORMAL = '\u001b[0m'

    print(BLUE)
    print(f'Correct: {len(correct)}')
    print(f'Wrong: {len(wrong)}')
    print(f'Errors: {len(errors)}')
    print(f'Accuracy: {round(len(correct) / len(list(test_dataset)), 2)}%')
    print(NORMAL)
except Exception as e:
    print(RED, 'Unknown error occurred while processing images:', NORMAL, e)
    exit()
