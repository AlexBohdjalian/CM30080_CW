from main import determine_angles
from generate_data import generate_test_data


# NOTE FOR MARKER: change this to the dir with images and list.txt
directory = 'Task1/assets/'
print('Using provided test dataset: ' + directory)

with open(f'{directory}list.txt', 'r') as f:
    results = f.readlines()
results = [(directory + str(item.split(',')[0]), int(item.split(',')[1].strip())) for item in results]

correct, wrong = determine_angles(results, show_output=False)

c_count = len(correct)
w_count = len(wrong)
e_count = len(results) - len(correct) - len(wrong)

BLUE = '\u001b[34m'
NORMAL = '\u001b[0m'

print(BLUE)
print(f'Correct: {c_count}')
print(f'Wrong: {w_count}')
print(f'Errors: {e_count}')
print(f'Accuracy: {round(c_count * 100 / len(results), 1)}%')
print(NORMAL)


test_random_data = True
if test_random_data:
    # Generate new random data for testing.
    import os
    import shutil

    try:
        print('Generating random test data...')
        test_dir = 'Task1/tmp_assets/'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)        
        os.makedirs(test_dir)

        file_names, actual_angles = generate_test_data(1000, test_dir)

        print('Evaluating test data...')
        correct, wrong = determine_angles(zip(file_names, actual_angles), show_output=False)

        c_count = len(correct)
        w_count = len(wrong)
        e_count = len(file_names) - len(correct) - len(wrong)

        print(BLUE)
        print(f'Correct: {c_count}')
        print(f'Wrong: {w_count}')
        print(f'Errors: {e_count}')
        print(f'Accuracy: {round(c_count * 100 / len(file_names), 1)}%')
        print(NORMAL)
    except:
        print('Unknown error occurred in ' + __name__ + ' while testing random data. Please try again')

    # for wrong_pred in wrong:
    #     if abs(wrong_pred[1] - wrong_pred[2]) <= 5:
    #         print(wrong_pred)

    # cleanup
    shutil.rmtree(test_dir)
