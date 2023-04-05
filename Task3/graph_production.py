import cv2
import numpy as np
import os
import shutil

from main import draw_text, read_training_dataset, read_test_dataset, remove_noise_from_image, feature_detection_marker


# Initial setup --------------------------------------------------------------------------
dir = 'Task3/report_assets'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

task3_no_rotation_images_dir = 'Task3/Task2Dataset/TestWithoutRotations/'
task3_rotated_images_dir = 'Task3/Task3Dataset/'
task3_training_data_dir = 'Task3/Task2Dataset/Training/'

all_no_rotation_images_and_features = read_test_dataset(task3_no_rotation_images_dir, '.txt', read_colour=True)
all_rotation_images_and_features = read_test_dataset(task3_rotated_images_dir, '.csv', read_colour=True)
all_training_images_and_paths = read_training_dataset(task3_training_data_dir)

best_params = {
    'BFMatcher': {
        'crossCheck': False,
        'normType': 2
    },
    'RANSAC': {
        'confidence': 0.9564838900729838,
        'maxIters': 1600,
        'ransacReprojThreshold': 5.53440270211734
    },
    'min_good_matches': 4,
    'ratioThreshold': 0.42352058295191136,
    'sift': {
        'contrastThreshold': 0.005457729696636313,
        'edgeThreshold': 11.188051836654086,
        'nOctaveLayers': 4,
        'nfeatures': 2100,
        'sigma': 1.8708988402771627
    }
}

sift = cv2.SIFT_create(**best_params['sift'])
bf = cv2.BFMatcher(**best_params['BFMatcher'])

# Get noise vs no noise image -----------------------------------------------------------
print('Creating noise removal example...')
gray_example_image = all_rotation_images_and_features[0][0]
no_noise_example_image = remove_noise_from_image(gray_example_image)

vis = np.concatenate((gray_example_image, no_noise_example_image), axis=1)
vis = cv2.line(vis, (round(vis.shape[1] / 2), 0), (round(vis.shape[1] / 2), vis.shape[0]), (0, 0, 0), 2)
vis = cv2.copyMakeBorder(vis, 0, 30, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
draw_text(vis, 'Before', True, pos=(round(vis.shape[0] * 0.5), vis.shape[0] - 15), font_scale=2, font_thickness=2 )
draw_text(vis, 'After', True, pos=(round(vis.shape[0] * 1.5), vis.shape[0] - 15), font_scale=2, font_thickness=2 )
cv2.imwrite(f'{dir}/noise_removal_example.png', vis)

# Get example of output with and without rotation
print('Creating example marker output...')
example_outputs = [
    ('bad_bounding_example_feature_detection', all_no_rotation_images_and_features[0]),
    ('no_rotation_example_feature_detection', all_no_rotation_images_and_features[6]),
    ('rotation_example_feature_detection', all_rotation_images_and_features[0])
]
for file_name, (gray_example_image, colour_example_image, actual_features) in example_outputs:
    img_copy = colour_example_image.copy()
    _, processed_image, _ = feature_detection_marker(
        sift,
        bf,
        gray_example_image,
        colour_example_image,
        all_training_images_and_paths,
        best_params,
        True
    )
    for name, bb_tl, bb_br in actual_features:
        cv2.rectangle(img_copy, bb_tl, bb_br, (0, 255, 0), 2)
        draw_text(
            img_copy,
            name,
            pos=bb_tl,
            text_color=(0, 255, 0),
            text_color_bg=(0, 0, 0)
        )
    vis = np.concatenate((processed_image, img_copy), axis=1)
    vis = cv2.line(vis, (round(vis.shape[1] / 2), 0), (round(vis.shape[1] / 2), vis.shape[0]), (0, 0, 0), 2)
    vis = cv2.copyMakeBorder(vis, 0, 30, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    draw_text(vis, 'Predicted', True, pos=(round(vis.shape[0] * 0.5), vis.shape[0] - 15), font_scale=2, font_thickness=2 , text_color_bg=(0, 0, 0))
    draw_text(vis, 'Actual', True, pos=(round(vis.shape[0] * 1.5), vis.shape[0] - 15), font_scale=2, font_thickness=2 , text_color_bg=(0, 0, 0))

    cv2.imwrite(f'{dir}/{file_name}.png', vis)

# Get graphs...
# Get example where it doesn't perform great

print('Done.')
