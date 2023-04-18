import shutil

import os

from main import determine_angles
from generate_data import generate_test_data


if __name__ == '__main__':

    params = {'CANNY': {'threshold1': 81,
                        'threshold2': 272,
                        'apertureSize': 3},
              'HOUGH': {'rho': 0.863647965611662,
                        'theta': 0.0008368119465516296,
                        'threshold': 37,
                        'minLineLength': 41.91973871274722,
                        'maxLineGap': 34.14817528989028},
              'DBSCAN': {'eps': 6, 'min_samples': 2}}

    # NOTE FOR MARKER: change this to the dir with images and list.txt
    directory = 'assets/'
    print('Using provided test dataset: ' + directory)

    with open(f'{directory}list.txt', 'r') as f:
        input_data = f.readlines()
    input_data = [(directory + str(item.split(',')[0]), int(item.split(',')[1].strip())) for item in input_data]

    results = determine_angles(input_data, params)

    correct = [item for item in results if item[1] == item[2]]
    wrong = [item for item in results if item[1] != item[2]]

    print(f'Correct: {len(correct)}')
    print(f'Wrong: {len(wrong)}')
    print(f'Accuracy: {len(correct) / len(results) * 100}%')


