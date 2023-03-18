import os
import time
import traceback

import cv2
import numpy as np

cv2.setRNGSeed(0)


BLUE = '\u001b[34m'
RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'


def main_process_for_marker(marker_test_dataset, training_dataset, params, show_output=False):
    try:
        sift = cv2.SIFT_create(**params['sift'])
        bf = cv2.BFMatcher(**params['BFMatcher'])

        correct = 0
        false_pos = 0
        false_neg = 0
        times = []
        for i in range(len(marker_test_dataset)):
            colour_query_image, actual_features = marker_test_dataset[i]

            print(f'Test Image {i} ...', end='')
            predicted_features, processed_image, run_time = feature_detection_marker(
                sift,
                bf,
                colour_query_image,
                training_dataset,
                params,
                show_output=show_output
            )

            if show_output:
                draw_text(processed_image, 'Press Q to see next...', font_scale=2, text_color=(255, 255, 255), text_color_bg=(0,0,0))
                draw_text(processed_image, 'Actual Bounding Box', pos=(0, 20), text_color=(0, 255, 0), font_scale=1, text_color_bg=(0,0,0))
                draw_text(processed_image, 'Predicted Bounding Box', pos=(0, 30), text_color=(255, 0, 0), font_scale=1, text_color_bg=(0,0,0))
                for name, bb_tl, bb_br in actual_features:
                    cv2.rectangle(processed_image, bb_tl, bb_br, (0, 255, 0), 2)
                    draw_text(processed_image, name, pos=bb_tl, text_color=(0, 255, 0), text_color_bg=(0, 0, 0))

                cv2.imshow('Test Image {i}', processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            times.append(run_time)
            predicted_feature_names_set = set([f[0] for f in predicted_features])
            actual_feature_names_set = set([f[0] for f in actual_features])

            false_pos_res = 'None'
            false_neg_res = 'None'
            if actual_feature_names_set == predicted_feature_names_set:
                correct += 1
                print(GREEN, 'Correct!!!', NORMAL)
            elif predicted_feature_names_set != actual_feature_names_set:
                false_pos_res = predicted_feature_names_set.difference(actual_feature_names_set)
                false_neg_res = actual_feature_names_set.difference(predicted_feature_names_set)
                false_pos += len(false_pos_res)
                false_neg += len(false_neg_res)
                print(RED, 'IN-Correct!!!', NORMAL)
            
            if isinstance(false_pos_res, set) and len(false_pos_res) == 0:
                false_pos_res = 'None'
            if isinstance(false_neg_res, set) and len(false_neg_res) == 0:
                false_neg_res = 'None'

            print('Predicted:', predicted_feature_names_set)
            print('Actual   :', actual_feature_names_set)
            print('False Positives:', false_pos_res)
            print('True Positives :', predicted_feature_names_set.intersection(actual_feature_names_set))
            print('False Negatives:', false_neg_res)
            print(f'Time to do feature matching: {round(run_time, 3)}ms')
            print()

        accuracy = correct * 100 / len(list(marker_test_dataset))
        total_false_results = false_pos + false_neg
        print(f'Accuracy: {accuracy}%')
        print(f'Total false results: {total_false_results}')
        print(f'Average runtime per image: {round(np.mean(times), 3)}ms')
        print(BLUE, f'Note: Total false results is the total number of false-positives and false-negatives for all test images', NORMAL)
        print(BLUE, f'Note: The average runtime is for the feature matching process '
              + 'and does not include any additional processing done to check '
              + 'accuracy, display the results, etc.', NORMAL)
    except:
        print(RED, 'Unknown error occurred:', NORMAL, traceback.format_exc())
        exit()



def feature_detection_hyperopt(sift, bf, query_image, all_training_data_kp_desc, params):
    query_kp, query_desc = sift.detectAndCompute(query_image, None)

    found_features = []
    for feature_path, current_kp, current_desc in all_training_data_kp_desc:
        try:
            matches = bf.knnMatch(query_desc, current_desc, k=2)
        except Exception as e:
            print(query_desc, current_desc)
            print(e)
            exit()

        # Get the best matches within a threshold distance
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < params['ratioThreshold'] * n.distance:
                    good_matches.append(m)

        if len(good_matches) >= 4:
            src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography using RANSAC and remove outliers
            M, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                **params['RANSAC'],
            )
            matches_mask = mask.ravel().tolist()
            inlier_matches = [good_matches[i] for i, val in enumerate(matches_mask) if val]

            if len(inlier_matches) > 0:
                feature_name = feature_name_from_path(feature_path)
                found_features.append((feature_name, (), ()))

    return found_features

def feature_detection_marker(sift, bf, colour_query_image, all_training_data, params, show_output=False):
    query_image = cv2.cvtColor(colour_query_image, cv2.COLOR_BGR2GRAY)
    
    extra_time = 0
    start_time = time.time()
    query_kp, query_desc = sift.detectAndCompute(query_image, None)

    found_features = []
    for feature_image, feature_image_path in all_training_data:
        current_kp, current_desc = sift.detectAndCompute(feature_image, None)

        matches = bf.knnMatch(query_desc, current_desc, k=2)

        # Get the best matches within a threshold distance
        good_matches = []
        for m, n in matches:
            if m.distance < params['ratioThreshold'] * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 4:
            src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography using RANSAC and remove outliers
            _, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                ransacReprojThreshold=params['RANSAC']['ransacReprojThreshold'],
                maxIters=params['RANSAC']['maxIters'],
                confidence=params['RANSAC']['confidence']
            )
            matches_mask = mask.ravel().tolist()
            inlier_matches = [good_matches[i] for i, val in enumerate(matches_mask) if val]

            if len(inlier_matches) > 0:
                feature_name = feature_name_from_path(feature_image_path)
                matched_query_kp = [query_kp[m.queryIdx] for m in good_matches]
                matched_query_xy = [(int(kp.pt[0]), int(kp.pt[1])) for kp in matched_query_kp]

                box = oriented_bounding_box(np.array(matched_query_xy))
                bb_top_left = np.min(box, axis=0)
                bb_bottom_right = np.max(box, axis=0)

                if show_output:
                    s_extra = time.time()
                    centroid = (
                        round((bb_top_left[0] + bb_bottom_right[0]) / 2),
                        round((bb_top_left[1] + bb_bottom_right[1]) / 2)
                    )
                    cv2.drawContours(colour_query_image, [box], 0, (255, 0, 0), 2)
                    draw_text(colour_query_image, text=feature_name, to_centre=True, pos=centroid, text_color=(255, 0, 0), text_color_bg=(0, 0, 0))
                    extra_time += time.time() - s_extra

                found_features.append((feature_name, bb_top_left, bb_bottom_right))
    run_time = time.time() - start_time - extra_time

    return found_features, colour_query_image, run_time

def read_test_dataset(dir, file_ext, read_colour=False):
    print(f'Reading test dataset: {dir}')
    colour_mode = cv2.IMREAD_GRAYSCALE
    if read_colour:
        colour_mode = cv2.IMREAD_COLOR

    image_files = os.listdir(dir + 'images/')
    image_files = sorted(image_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

    all_data = []
    for image_file in image_files:
        csv_file = dir + 'annotations/' + image_file[:-4] + file_ext
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
        path = dir + 'images/' + image_file

        all_data.append((
            cv2.imread(path, colour_mode),
            all_features
        ))

    return all_data

def read_training_dataset(dir):
    print(f'Reading training dataset: {dir}')
    return [(
        cv2.imread(dir + 'png/' + path, cv2.IMREAD_GRAYSCALE),
        path
    ) for path in os.listdir(dir + 'png/')]

def feature_name_from_path(img_path):
    return img_path[img_path.find('-')+1:img_path.find('.png')]

def oriented_bounding_box(points):
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    return np.int0(box)

def draw_text(img, text,
          to_centre=False,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(255, 255, 255),
          text_color_bg=(0, 255, 0)
          ):
    """ Draws text on a cv2 image in a given spot with a background """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if to_centre:
        x -= text_w // 2
        y -= text_h // 2
    cv2.rectangle(img, (x, y), (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size
