from main import determine_angles
from generate_data import generate_test_data

# NOTE FOR MARKER: change this to the dir with images and list.txt
directory = 'Task1/assets/'

with open(f'{directory}list.txt', 'r') as f:
    results = f.readlines()
results = [(directory + str(item.split(',')[0]), int(item.split(',')[1].strip())) for item in results]

determine_angles(results)

test_random_data = True
if test_random_data:
    # Generate new random data for testing.
    import os
    import shutil

    print('Generating random test data...')
    test_dir = 'Task1/tmp_assets/'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    file_names, actual_angles = generate_test_data(30, test_dir)

    print('Evaluating test data...')
    correct, wrong = determine_angles(zip(file_names, actual_angles), show_output=False)

    BLUE = '\u001b[34m'
    NORMAL = '\u001b[0m'

    c_count = len(correct)
    w_count = len(wrong)
    e_count = len(file_names) - len(correct) - len(wrong)

    print(BLUE)
    print(f'Correct: {c_count}')
    print(f'Wrong: {w_count}')
    print(f'Errors: {e_count}')
    print(f'Accuracy: {round(c_count * 100 / len(file_names), 1)}%')
    print(NORMAL)

    # cleanup
    shutil.rmtree(test_dir)
