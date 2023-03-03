import os
import cv2
import shutil
from main import training_process, feature_detection

BLUE = '\u001b[34m'
RED = '\u001b[31m'
NORMAL = '\u001b[0m'

task2_no_rotation_images_dir = 'Task2/Task2Dataset/TestWithoutRotations/'
task2_rotated_images_dir = 'Task2/Task3Dataset/'
task2_training_data_dir = 'Task2/Task2Dataset/Training/'

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
    all_no_rotation_images_and_features = read_no_rotations_dataset(task2_no_rotation_images_dir)
    all_training_data = read_training_dataset(task2_training_data_dir)
    all_rotation_images_and_features = read_rotations_dataset(task2_rotated_images_dir)
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, e)
    print(NORMAL)
    exit()

directions = 16
angles = [360 / directions * i for i in range(directions)] # up, tr, right, br, down, bl, left, tl
tmp_file_dir = 'TmpRotated/'
rotated_training_dataset = []
print('Applying rotation to training set ...')
try:
    if os.path.exists(task2_training_data_dir + tmp_file_dir):
        shutil.rmtree(task2_training_data_dir + tmp_file_dir)
    os.makedirs(task2_training_data_dir + tmp_file_dir)

    for training_image in all_training_data:
        image = cv2.imread(training_image)
        for angle in angles:
            height, width = image.shape[:2] # image shape has 3 dimensions
            image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

            rotated_image = cv2.getRotationMatrix2D(image_center, angle, 1.)

            # rotation calculates the cos and sin, taking absolutes of those.
            abs_cos = abs(rotated_image[0,0]) 
            abs_sin = abs(rotated_image[0,1])

            # find the new width and height bounds
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            # subtract old image center (bringing image back to origo) and adding the new image center coordinates
            rotated_image[0, 2] += bound_w/2 - image_center[0]
            rotated_image[1, 2] += bound_h/2 - image_center[1]

            # rotate image with the new bounds and translated rotation matrix
            rotated_image = cv2.warpAffine(image, rotated_image, (bound_w, bound_h), borderValue=(255,255,255))

            rotated_image_name = training_image.replace('/Training/png/', f'/Training/{tmp_file_dir}{angle}deg_')
            cv2.imwrite(rotated_image_name, rotated_image)
            rotated_training_dataset.append(rotated_image_name)
except Exception as e:
    if os.path.exists(task2_training_data_dir + tmp_file_dir):
        shutil.rmtree(task2_training_data_dir + tmp_file_dir)
    print(RED, 'Error while applying rotation to training dataset:', NORMAL, e)
    exit()

print('Training...')
try:
    training_process(all_training_data)
except Exception as e:
    print(RED, 'Unknown error occurred while processing images:', NORMAL, e)
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
            predicted_features = feature_detection(image_path)

            if predicted_features == actual_features:
                correct.append(image_path)
            else:
                wrong.append(image_path)
        except:
            print(RED, 'Uncaught error occurred while testing image:', image_path, NORMAL)
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
