import traceback

from main import main_process_for_marker, read_test_dataset, read_training_dataset, draw_text


RED = '\u001b[31m'
NORMAL = '\u001b[0m'

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
        print()
    except Exception as e:
        print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
        exit()

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    main_process_for_marker(test_dataset, all_training_data, params, show_output=True)
