import os
from main import feature_detection

test_dir = 'Task2/Task2Dataset/TestWithoutRotations/'

def read_test_dir(test_dir):
    print(f'Reading dataset {test_dir} ...')
    image_files = os.listdir(test_dir + 'images/')
    image_files = sorted(image_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

    actual_image_features = []

    for image_file in image_files:
        csv_file = test_dir + 'annotations/' + image_file[:-4] + '.txt'
        with open(csv_file, 'r') as fr:
            features = fr.read().splitlines()
        for feature in features:
            end_of_class = feature.find(", ")
            end_of_tuple = feature.find("), (") + 1
            feature_class = feature[:end_of_class]
            feature_coord1 = eval(feature[end_of_class + 2:end_of_tuple])
            feature_coord2 = eval(feature[end_of_tuple + 2:])

            actual_image_features.append([feature_class, feature_coord1, feature_coord2])

    return image_files, actual_image_features

all_images, all_features = read_test_dir(test_dir)

print('Processing images...')
correct = []
wrong = []
errors = []
for image_path, actual_features in zip(all_images, all_features):
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
print(f'Accuracy: {round(len(correct) / len(all_images), 2)}%')
print(NORMAL)