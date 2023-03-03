import os
import shutil
import time
import cv2
import numpy as np
import traceback
from main import feature_detection


BLUE = '\u001b[34m'
RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

task3_no_rotation_images_dir = 'Task3/Task2Dataset/TestWithoutRotations/'
task3_rotated_images_dir = 'Task3/Task3Dataset/'
task3_training_data_dir = 'Task3/Task2Dataset/Training/'

def read_no_rotations_dataset(dir):
    print(f'Reading dataset with no rotated images: {dir}')
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

def read_rotations_dataset(dir):
    print(f'Reading dataset with rotated images: {dir}')
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

def read_training_dataset(dir):
    print(f'Reading dataset with training images: {dir}')
    return [dir + 'png/' + path for path in os.listdir(dir + 'png/')]

try:
    all_no_rotation_images_and_features = read_no_rotations_dataset(task3_no_rotation_images_dir)
    all_rotation_images_and_features = read_rotations_dataset(task3_rotated_images_dir)
    all_training_data = read_training_dataset(task3_training_data_dir)
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
    exit()

directions = 8
# TODO: might need to increase this
angles = [360 / directions * i for i in range(directions)] # up, tr, right, br, down, bl, left, tl
tmp_file_dir = 'TmpRotated/'
rotated_training_dataset = []
print('Applying rotation to training set ...')
try:
    if os.path.exists(task3_training_data_dir + tmp_file_dir):
        shutil.rmtree(task3_training_data_dir + tmp_file_dir)
    os.makedirs(task3_training_data_dir + tmp_file_dir)

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

            rotated_image_name = training_image.replace('/Training/png/', f'/Training/{tmp_file_dir}{round(angle)}deg_')
            cv2.imwrite(rotated_image_name, rotated_image)
            rotated_training_dataset.append(rotated_image_name)
except Exception as e:
    if os.path.exists(task3_training_data_dir + tmp_file_dir):
        shutil.rmtree(task3_training_data_dir + tmp_file_dir)
    print(RED, 'Error while applying rotation to training dataset:', NORMAL, traceback.format_exc())
    exit()

all_training_data = rotated_training_dataset

try:
    params_list = [{'nfeatures': nf, 'nOctaveLayers': nl, 'contrastThreshold': ct, 'edgeThreshold': et, 'sigma': s, 'matchThreshold': mf}
        for nf in [0]
        for nl in [2, 3, 4, 5]
        for ct in [0.01, 0.05, 0.09]
        for et in [5, 10, 15, 20]
        for s in [1.2, 1.6, 2.0]
        # TODO: re-run with matchThreshold parameter as well now
        for mf in np.arange(45.0, 55.0, 0.25)
    ]
    # with rotated training dataset:
    # no rotations: 90% acc, 2 false pos, 0 false neg
    # rotations   : 40% acc
    # both        : 65% acc, 14 false pos, 1 false neg
    # observations: if nOctaveLayers is anything but 2 it performs quite bad, 
    params_list = [{'nfeatures': 0, 'nOctaveLayers': 2, 'contrastThreshold': 0.01, 'edgeThreshold': 15, 'sigma': 2.0, 'matchThreshold': 50}]

    best_acc = 0
    best_params = {}
    start_t = time.time()
    test_dataset = all_no_rotation_images_and_features# + all_rotation_images_and_features
    print(f'There are {len(test_dataset)} images to classify ...')
    for i in range(len(params_list)):
        print(f'Grid-search progress: {i + 1}/{len(params_list)}\t...', end='\r')
        correct = 0
        false_pos = 0
        false_neg = 0
        for image_path, actual_features in test_dataset: # replace this with suitable dataset (or combined)
            predicted_features = sorted(feature_detection(image_path, all_training_data, params_list[i]), key=lambda x:x[0])
            # used = []
            # predicted_features = [[sub, used.append(sub[0])][0] for sub in predicted_features if sub[0] not in used]
            print('For:', image_path)
            print('Predicted:', [feature[0] for feature in predicted_features])
            print('Actual   :', [feature[0] for feature in actual_features])

            if len(predicted_features) == len(actual_features) and all(f1[0] == f2[0] for f1, f2 in zip(predicted_features, actual_features)):
                correct += 1
                print(GREEN, 'Correct!!!', NORMAL)
            else:
                false_pos_res = [x[0] for x in predicted_features if x[0] not in [s[0] for s in actual_features]]
                false_neg_res = [x[0] for x in actual_features if x[0] not in [s[0] for s in predicted_features]]
                # print(RED, 'False pos:', false_pos, NORMAL)
                # print(RED, 'False neg:', false_neg, NORMAL)
                false_pos += any(false_pos_res)
                false_neg += any(false_neg_res)
                print(RED, 'IN-Correct!!!', NORMAL)

        accuracy = correct * 100 / len(list(test_dataset))
        if accuracy > best_acc:
            best_acc = accuracy
            best_params = params_list[i]

        print(f'Grid-search progress: {i + 1}/{len(params_list)}\tFalse Pos: {false_pos}\tFalse Neg: {false_neg}\tAccuracy: {round(accuracy, 1)}%\tTime: {round(time.time() - start_t)}s')

        # print(BLUE)
        # print(f'Correct: {len(correct)}')
        # print(f'Wrong: {len(wrong)}')
        # print(f'Accuracy: {round(accuracy, 1)}%')
        # print(NORMAL)

    print(BLUE, f'Best Params: {best_params}', NORMAL)
    print(BLUE, f'Best Params accuracy: {best_acc}%', NORMAL)
except Exception as e:
    print(RED, 'Unknown error occurred while processing images:', NORMAL, traceback.format_exc())
    exit()

# # Tidy-up
# if os.path.exists(task3_training_data_dir + tmp_file_dir):
#     shutil.rmtree(task3_training_data_dir + tmp_file_dir)
