import os
import shutil
import traceback

import cv2
from main import feature_detection_for_graphing, read_test_dataset, read_training_dataset
# from main import draw_text, feature_detection_marker, read_test_dataset, read_training_dataset, remove_noise_from_image
from matplotlib import pyplot as plt


# Initial setup --------------------------------------------------------------------------
dir = 'Task3/report_assets'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

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
    print('Error while reading datasets:', traceback.format_exc())
    exit()

# best_params = {
#     'BFMatcher': {
#         'crossCheck': False,
#         'normType': 2
#     },
#     'RANSAC': {
#         'confidence': 0.9564838900729838,
#         'maxIters': 1600,
#         'ransacReprojThreshold': 5.53440270211734
#     },
#     'min_good_matches': 4,
#     'ratioThreshold': 0.42352058295191136,
#     'sift': {
#         'contrastThreshold': 0.005457729696636313,
#         'edgeThreshold': 11.188051836654086,
#         'nOctaveLayers': 4,
#         'nfeatures': 2100,
#         'sigma': 1.8708988402771627
#     }
# }
best_params = {
    'RANSAC': {
        'ransacReprojThreshold': 11.110294305510669
    },
    'SIFT': {
        'contrastThreshold': 0.0039052330228148877,
        'edgeThreshold': 16.379139206562137,
        'nOctaveLayers': 6,
        'nfeatures': 1700,
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

sift = cv2.SIFT_create(**best_params['SIFT'])
bf = cv2.BFMatcher(**best_params['BF'])


# Determine "Accuracy vs Parameter", "Mean Average Error vs Parameter & Average time per image vs Parameter"  -----------------------------------------------------------
def create_subplots(field, var, accuracies, fps_and_tps, times):
    _, axs = plt.subplots(1, 3, figsize=(14, 5))

    # Plot accuracy
    axs[0].plot(var, accuracies, label='Accuracy')
    axs[0].set_xlabel(field)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title(f'Accuracy vs. {field}')

    # Plot False Positives and True Positives
    false_pos = [i[0] for i in fps_and_tps]
    true_pos = [i[1] for i in fps_and_tps]
    axs[1].plot(var, false_pos, color='blue')
    axs[1].set_xlabel(field)
    axs[1].set_ylabel('False Positives', color='blue')
    axs[1] = axs[1].twinx()
    axs[1].plot(var, true_pos, color='yellow')
    axs[1].set_ylabel('True Positives', color='yellow')
    axs[1].set_title(f'False/True Positives vs. {field}')

    # Plot Time
    axs[2].plot(var, times, label='Time')
    axs[2].set_xlabel(field)
    axs[2].set_ylabel('Time taken per image')
    axs[2].set_title(f'Time taken per image vs. {field}')


    return axs


def get_sensitivity(field_name, param_space, values):
    accuracies = []
    fps_and_tps = []
    times = []

    for param in param_space:
        params = best_params.copy()
        for key, val in param.items():
            if key in params:
                if isinstance(val, dict):
                    params[key] = {**params[key], **val}
                else:
                    params[key] = val

        accuracy, false_positives, true_positives, avg_time_per_image = feature_detection_for_graphing(
            train_data,
            query_data,
            params
        )

        accuracies.append(accuracy)
        fps_and_tps.append((false_positives, true_positives))
        times.append(avg_time_per_image)

    return create_subplots(field_name, values, accuracies, fps_and_tps, times)


param_spaces_to_try = [
    ('nOctaveLayers', [{ 'SIFT': { 'nOctaveLayers': x }} for x in range(1, 8)], list(range(1, 8))),
    ('nfeatures', [{'SIFT': { 'nfeatures': x }} for x in range(0, 3000, 100)], list(range(0, 3000, 100))),
]

for param_name, param_space, values in param_spaces_to_try:
    get_sensitivity(param_name, param_space, values)
    plt.savefig(f'{dir}/{param_name}_plot.png')


# # Get noise vs no noise image -----------------------------------------------------------
# print('Creating noise removal example...')
# gray_example_image = all_rotation_images_and_features[0][0]
# no_noise_example_image = remove_noise_from_image(gray_example_image)

# vis = np.concatenate((gray_example_image, no_noise_example_image), axis=1)
# vis = cv2.line(vis, (round(vis.shape[1] / 2), 0), (round(vis.shape[1] / 2), vis.shape[0]), (0, 0, 0), 2)
# vis = cv2.copyMakeBorder(vis, 0, 30, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
# draw_text(vis, 'Before', True, pos=(round(vis.shape[0] * 0.5), vis.shape[0] - 15), font_scale=2, font_thickness=2 )
# draw_text(vis, 'After', True, pos=(round(vis.shape[0] * 1.5), vis.shape[0] - 15), font_scale=2, font_thickness=2 )
# cv2.imwrite(f'{dir}/noise_removal_example.png', vis)

# # Get example of output with and without rotation
# print('Creating example marker output...')
# example_outputs = [
#     ('bad_bounding_example_feature_detection', all_no_rotation_images_and_features[0]),
#     ('no_rotation_example_feature_detection', all_no_rotation_images_and_features[6]),
#     ('rotation_example_feature_detection', all_rotation_images_and_features[0])
# ]
# for file_name, (gray_example_image, colour_example_image, actual_features) in example_outputs:
#     img_copy = colour_example_image.copy()
#     _, processed_image, _ = feature_detection_marker(
#         sift,
#         bf,
#         gray_example_image,
#         colour_example_image,
#         all_training_images_and_paths,
#         best_params,
#         True
#     )
#     for name, bb_tl, bb_br in actual_features:
#         cv2.rectangle(img_copy, bb_tl, bb_br, (0, 255, 0), 2)
#         draw_text(
#             img_copy,
#             name,
#             pos=bb_tl,
#             text_color=(0, 255, 0),
#             text_color_bg=(0, 0, 0)
#         )
#     vis = np.concatenate((processed_image, img_copy), axis=1)
#     vis = cv2.line(vis, (round(vis.shape[1] / 2), 0), (round(vis.shape[1] / 2), vis.shape[0]), (0, 0, 0), 2)
#     vis = cv2.copyMakeBorder(vis, 0, 30, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
#     draw_text(vis, 'Predicted', True, pos=(round(vis.shape[0] * 0.5), vis.shape[0] - 15), font_scale=2, font_thickness=2 , text_color_bg=(0, 0, 0))
#     draw_text(vis, 'Actual', True, pos=(round(vis.shape[0] * 1.5), vis.shape[0] - 15), font_scale=2, font_thickness=2 , text_color_bg=(0, 0, 0))

#     cv2.imwrite(f'{dir}/{file_name}.png', vis)

# # Get graphs...
# # Get example where it doesn't perform great

print('Done.')
