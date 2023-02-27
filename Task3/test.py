import os
from main import training_process, feature_detection

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
        all_data.append([image_file, all_features])

    return all_data

def read_training_dataset(dir):
    print(f'Reading dataset {dir} ...')
    return list(os.listdir(dir + 'png/'))

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
        all_data.append([image_file, all_features])

    return all_data

try:
    all_no_rotation_images_and_features = read_no_rotations_dataset(task3_task2Dataset_dir + 'TestWithoutRotations/')
    all_training_data = read_training_dataset(task3_task2Dataset_dir + 'Training/')
    all_rotation_images_and_features = read_rotations_dataset(task3_task3Dataset_dir)
except Exception as e:
    print('Error while reading datasets:', e)
    print()
    exit()

print('Training...')
try:
    training_process(all_training_data)
except Exception as e:
    print('Unknown error occurred while processing images:', e)
    exit()


print('Processing images...')
try:
    correct = []
    wrong = []
    errors = []
    test_dataset = all_no_rotation_images_and_features
    for image_path, actual_features in test_dataset: # replace this with suitable dataset (or combined)
        try:
            predicted_features = feature_detection(image_path)

            if predicted_features == actual_features:
                correct.append(image_path)
            else:
                wrong.append(image_path)
        except:
            print('Uncaught error occurred while testing image:', image_path)
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
    print(NORMAL)
    print('Unknown error occurred while processing images:', e)
    exit()
