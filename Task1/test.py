from main import determine_angles

# NOTE FOR MARKER: change this to the dir with images and list.txt
directory = 'Task1/assets/'

with open(f'{directory}list.txt', 'r') as f:
    results = f.readlines()
results = [(directory + str(item.split(',')[0]), int(item.split(',')[1].strip())) for item in results]

determine_angles(results)

test_random_data = True
if test_random_data:
    # Generate new random data for testing.
    import numpy as np
    import cv2
    import os
    import shutil


    test_dir = 'Task1/tmp_assets/'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Define image dimensions
    height = 480
    width = 854

    file_names = []
    actual_angles = []

    print('Generating random test data...')

    for i in range(10):
        # Create a blank gray image
        image = np.zeros((height, width, 3), np.uint8)
        image[:, :] = (51, 51, 51)

        # Define line properties
        line_length = np.random.randint(30, 100)
        line_thickness = 2
        line_color = (255, 255, 255)

        # Generate a random start point that is within the image bounds
        start_point = (
            np.random.randint(line_length, width - line_length),
            np.random.randint(line_length, height - line_length)
        )

        # Generate two random directions for the lines
        angles = np.random.rand(2) * 2 * np.pi

        # Calculate the end points of the lines using the start point and random directions
        end_points = []
        for j in range(2):
            # Calculate the maximum distance the line can travel in the given direction
            max_distance = min(line_length, min(start_point[0], width - start_point[0],
                                                start_point[1], height - start_point[1]))
            # Calculate the end point of the line using the maximum distance
            end_point = (int(start_point[0] + max_distance * np.cos(angles[j])),
                        int(start_point[1] + max_distance * np.sin(angles[j])))
            end_points.append(end_point)

        # Draw the lines on the image
        for end_point in end_points:
            cv2.line(image, start_point, end_point, line_color, line_thickness)

        # Calculate the actual angle between the two lines
        vector1 = np.array(end_points[0]) - np.array(start_point)
        vector2 = np.array(end_points[1]) - np.array(start_point)
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cos_angle)
        angle_degrees = round(np.rad2deg(angle))

        # Save the image
        img_path = test_dir + 'image' + str(i + 1) + '.png'
        # cv2.imshow('image', image)
        # cv2.waitKey(1000)
        cv2.imwrite(img_path, image)
        file_names.append(img_path)
        actual_angles.append(angle_degrees)

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
