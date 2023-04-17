import traceback

from main import feature_detection, read_test_dataset, read_training_dataset

RED = '\u001b[31m'
NORMAL = '\u001b[0m'


if __name__ == '__main__':
    params = {
        'RANSAC': {
            'ransacReprojThreshold': 11.110294305510669
        },
        'SIFT': {
            'contrastThreshold': 0.0039052330228148877,
            'edgeThreshold': 16.379139206562137,
            'nOctaveLayers': 6,
            'nfeatures': 1700, # NOTE: faster with 1300 on given training data
            'sigma': 2.2201211013686857
        },
        'BF': {
            'crossCheck': False,
            'normType': 2
        },
        'inlierScore': 4,
        'ratioThreshold': 0.6514343913409797,
        'resizeQuery': 95
    }

    # NOTE: For marker, we have assumed that the additional data you have is in the same format as the data given.
    # Please replace the below three directories with your own and then execute this file.
    train_data_dir = 'Task3/Task2Dataset/Training/'
    query_data_dirs = [
        ('Task3/Task3Dataset/', '.csv'),
        ('Task3/Task2Dataset/TestWithoutRotations/', '.txt'),
    ]

    try:
        train_data = read_training_dataset(train_data_dir)
        query_data = []
        for q_dir, ext in query_data_dirs:
            for data in read_test_dataset(q_dir, file_ext=ext):
                query_data.append(data)
    except Exception as e:
        print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
        exit()

    # set the show_output to True to see the query images with the detected features
    # if set to true ignore 'Avg. time per query image' metric
    feature_detection(train_data, query_data, params, show_output=True)
