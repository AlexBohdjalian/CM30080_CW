import os
import cv2
import time
import traceback
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

cv2.setRNGSeed(0)
n_workers = cv2.getNumberOfCPUs()

# Define constants
BLUE = '\u001b[34m'
RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'
image_info = (
    ('Press Q to see next...', (0, 0), 2, (255, 255, 255)),
    ('Actual Bounding Box', (0, 20), 1, (0, 255, 0)),
    ('Predicted Bounding Box', (0, 30), 1, (255, 0, 0))
)


def filter_covered_boxes(bounding_boxes):
    filtered_boxes = []

    for i1, ((x1, y1, x2, y2), bbx1) in enumerate(bounding_boxes):
        larger = True
        for i2, ((x3, y3, x4, y4), bbx2) in enumerate(bounding_boxes):
            if i1 == i2:
                continue
            if x3 <= x1 and x4 >= x2 and y3 <= y1 and y4 >= y2:
                larger = False
                break
        if larger:
            filtered_boxes.append(((x1, y1, x2, y2), bbx1))
    return filtered_boxes


def scale_dimensions(dimension, scale):
    return round(dimension/scale)


def add_border(x_min, y_min, x_max, y_max, width, height, border=1):
    x_min = max(x_min-border, 0)
    y_min = max(y_min-border, 0)
    x_max = min(x_max+border, height)
    y_max = min(y_max+border, width)
    return x_min, y_min, x_max, y_max


def get_orientated_bounding_box(contour, scale=10):
    min_rect = cv2.minAreaRect(contour)
    (center, size, angle) = min_rect

    # Scale the min_rect back to the original image size
    center = tuple(np.array(center) / scale)
    size = tuple(np.array(size) / scale)

    # Create a new min_rect with the scaled values
    scaled_min_rect = (center, size, angle)

    # Convert the min_rect to a 4-point bounding box
    box = cv2.boxPoints(scaled_min_rect)
    box = np.int0(box)
    return box


def get_bounding_boxes(image, scale=10, min_area=500):
    bounding_boxes = []
    width, height = image.shape[:2]
    # Resize the image by the scaling factor as it improves the accuracy of the contour detection
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # Binarise and remove noise from the image
    image = cv2.medianBlur(image, 25)
    _, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((11, 11), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = [scale_dimensions(elem, scale) for elem in cv2.boundingRect(c)]
        # Ignore small contours that are likely to be noise
        if w * h < min_area:
            continue
        x_min, y_min, x_max, y_max = add_border(x, y, x+w, y+h, width, height)
        bounding_boxes.append(((x_min, y_min, x_max, y_max), get_orientated_bounding_box(c)))
    return bounding_boxes


def segment_icons(image):
    image_segments = []
    # get bounding boxes for the image
    bounding_boxes = get_bounding_boxes(image.copy())
    # filter out boxes that are entirely covered by a larger bounding box
    bounding_boxes = filter_covered_boxes(bounding_boxes)

    for (x_min, y_min, x_max, y_max), bbx in bounding_boxes:
        image_segment = image[y_min:y_max, x_min:x_max].copy()
        border = 5
        padded_image = cv2.copyMakeBorder(
            image_segment,
            top=border,
            bottom=border,
            left=border,
            right=border,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        image_segments.append((padded_image, bbx))
    return image_segments


def sift_detect_and_compute(sift_params, image, resize=None, return_vars=None):
    # calculate keypoints and descriptors for the image (resize if necessary)
    sift = cv2.SIFT_create(**sift_params)
    if resize:
        image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_LINEAR)
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc, return_vars


def segment_detect_and_compute(sift_params, image, resize, return_vars=None):
    segments = segment_icons(image.copy())
    segments_kp_desc = []
    # calculate keypoins and descriptors for each segment (i.e. individual icon in the image)
    for image, bounding_box in segments:
        kp, desc, _ = sift_detect_and_compute(sift_params, image, resize)
        if desc is not None:
            segments_kp_desc.append(((kp, desc), image, bounding_box))
    return segments_kp_desc, return_vars


def draw_bounding_box(image, bounding_box, text, colour=(0, 255, 0)):
    # draw icons bounding box and predicted name
    cv2.drawContours(image, [bounding_box], 0, colour, 2)
    # Find the highest point (minimum y-coordinate)
    highest_point = min(bounding_box, key=lambda pt: pt[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (highest_point[0], highest_point[1] - 5)
    cv2.putText(image, text, text_position, font, 0.5, colour, 1, cv2.LINE_AA)


def feature_detection(training_data, query_data, params, show_output=True):
    start_time = time.time()
    extra_time = 0

    bf = cv2.BFMatcher(**params['BF'])

    # compute keypoints and descriptors for training data
    with ThreadPoolExecutor(max_workers=n_workers) as executor1:
        futures_train = [executor1.submit(
            sift_detect_and_compute,
            params['SIFT'],
            train_image,
            None,
            feature
        ) for train_image, feature in training_data]

    # collect threading futures and filter out any images that have no descriptors
    all_training_data_kp_desc = []
    for future in futures_train:
        train_kp, train_desc, feature = future.result()
        if train_desc is None:
            print(f'No descriptors found for {feature}')
            continue
        all_training_data_kp_desc.append((feature, train_kp, train_desc))

    # segment query images into individual icons and computer keypoints and descriptors
    with ThreadPoolExecutor(max_workers=n_workers) as executor3:
        futures_query = [executor3.submit(
            segment_detect_and_compute,
            params['SIFT'],
            query_image,
            params['resizeQuery'],
            (path, actual_features)
        ) for path, query_image, actual_features in query_data]

    total_results = 0
    correct_results = 0
    false_positives = []
    false_negatives = []
    # main loop to match query images to training data
    for future in futures_query:
        segments_kp_desc, (path, actual_features) = future.result()
        if len(segments_kp_desc) == 0:
            print(f'No descriptors found for {path}')
            continue

        query_image = cv2.imread(path)

        predicted_features = []
        for (query_kp, query_desc), segment, bounding_box in segments_kp_desc:
            for feature_name, train_kp, train_desc in all_training_data_kp_desc:
                matches = bf.knnMatch(query_desc, train_desc, k=2)

                # filter out matches using Lowe's ratio test
                good_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < params['ratioThreshold'] * n.distance:
                            good_matches.append(m)

                # at least 4 matches are needed for homography
                if len(good_matches) < 4:
                    continue

                # Extract source (query) and destination (train) keypoints coordinates from good matches
                src_pts = np.float32([train_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography matrix between source and destination points using RANSAC
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, **params['RANSAC'])
                matches_mask = mask.ravel().tolist()

                # Check if the match has more than 'inlierScore' inliers
                if sum(matches_mask) > params['inlierScore']:
                    if show_output:
                        extra_time -= time.time()
                        draw_bounding_box(query_image, bounding_box, feature_name)
                        extra_time += time.time()
                    predicted_features.append(feature_name)
                    correct_results += 1
            total_results += 1

        predicted_feature_names_set = set(predicted_features)
        actual_feature_names_set = set([f[0] for f in actual_features])

        # calculate accuracy
        correct_predictions = predicted_feature_names_set.intersection(actual_feature_names_set)

        # calculate false positives and false negatives
        extra_time -= time.time()
        false_neg_diff = actual_feature_names_set.difference(predicted_feature_names_set)
        false_negatives += list(false_neg_diff)
        extra_time += time.time()

        false_pos_diff = predicted_feature_names_set.difference(actual_feature_names_set)
        false_positives += list(false_pos_diff)

        accuracy = round(len(correct_predictions) / len(actual_feature_names_set) * 100, 1)
        if accuracy == 100 and len(false_positives) == 0:
            print(GREEN, f"{path} -> Accuracy: {accuracy}%", NORMAL)
        else:
            print(RED, f"{path} -> Accuracy: {accuracy}%, True Positives: {correct_predictions}, False Positives: {false_pos_diff}, False Negatives {false_neg_diff}", NORMAL)

        if show_output:
            extra_time -= time.time()
            cv2.imshow('image', query_image)
            cv2.waitKey(0)
            extra_time += time.time()

    end_time = time.time()
    avg_time_per_image = round((end_time - start_time - extra_time) / len(query_data), 3)

    print('\nSummary of results:')
    print(f'False positives: {Counter(false_positives).most_common()}')
    print(f'False negatives: {Counter(false_negatives).most_common()}')
    print(f'Final accuracy: {round(correct_results/total_results * 100, 2)}%')
    print(f'Avg. time per query image: {avg_time_per_image} seconds')

    if show_output:
        cv2.destroyAllWindows()


def feature_detection_for_graphing(training_data, query_data, params):
    start_time = time.time()

    bf = cv2.BFMatcher(**params['BF'])

    # compute keypoints and descriptors for training data
    with ThreadPoolExecutor(max_workers=n_workers) as executor1:
        futures_train = [executor1.submit(
            sift_detect_and_compute,
            params['SIFT'],
            train_image,
            None,
            feature
        ) for train_image, feature in training_data]

    # collect threading futures and filter out any images that have no descriptors
    all_training_data_kp_desc = []
    for future in futures_train:
        train_kp, train_desc, feature = future.result()
        if train_desc is None:
            continue
        all_training_data_kp_desc.append((feature, train_kp, train_desc))

    # segment query images into individual icons and computer keypoints and descriptors
    with ThreadPoolExecutor(max_workers=n_workers) as executor3:
        futures_query = [executor3.submit(
            segment_detect_and_compute,
            params['SIFT'],
            query_image,
            params['resizeQuery'],
            (path, actual_features)
        ) for path, query_image, actual_features in query_data]

    accuracies = []
    true_positives = 0
    false_positives = []
    # main loop to match query images to training data
    for future in futures_query:
        segments_kp_desc, (path, actual_features) = future.result()
        if len(segments_kp_desc) == 0:
            continue

        predicted_features = []
        for (query_kp, query_desc), _, _ in segments_kp_desc:
            for feature_name, train_kp, train_desc in all_training_data_kp_desc:
                matches = bf.knnMatch(query_desc, train_desc, k=2)

                # filter out matches using Lowe's ratio test
                good_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < params['ratioThreshold'] * n.distance:
                            good_matches.append(m)

                # at least 4 matches are needed for homography
                if len(good_matches) < 4:
                    continue

                # Extract source (query) and destination (train) keypoints coordinates from good matches
                src_pts = np.float32([train_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography matrix between source and destination points using RANSAC
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, **params['RANSAC'])
                matches_mask = mask.ravel().tolist()

                # Check if the match has more than 'inlierScore' inliers
                if sum(matches_mask) > params['inlierScore']:
                    predicted_features.append(feature_name)

        predicted_feature_names_set = set(predicted_features)
        actual_feature_names_set = set([f[0] for f in actual_features])

        # calculate accuracy
        correct_predictions = predicted_feature_names_set.intersection(actual_feature_names_set)

        # calculate false positives and false negatives
        false_pos_diff = predicted_feature_names_set.difference(actual_feature_names_set)
        false_positives += list(false_pos_diff)
        true_positives += len(correct_predictions)

        accuracies.append(round(len(correct_predictions) / len(actual_feature_names_set) * 100, 1))

    end_time = time.time()
    avg_time_per_image = round((end_time - start_time) / len(query_data), 3)

    return np.mean(accuracies), len(false_positives), true_positives, avg_time_per_image 


# TODO: remove?
def feature_detection_hyperopt(bf, kp_desc_query, train_kp, train_desc, feature_name, params):

    for (query_kp, query_desc), _ in kp_desc_query:

        matches = bf.knnMatch(query_desc, train_desc, k=2)

        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < params['ratioThreshold'] * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            continue

        src_pts = np.float32([train_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, **params['RANSAC'])
        matches_mask = mask.ravel().tolist()

        if sum(matches_mask) > params['inlierScore']:
            return feature_name

    return

# TODO: remove?
def main_process_for_marker(test_images_and_features, training_images_and_paths, params, show_output=False):
    try:
        # Create the sift and matcher objects once at the start for efficiency
        sift = cv2.SIFT_create(**params['sift'])
        bf = cv2.BFMatcher(**params['BFMatcher'])

        correct = 0
        false_pos = 0
        false_neg = 0
        times = []
        # For each image, get the predicted features and check how accurate they are
        for i, (gray_query_image, colour_query_image, actual_features) in enumerate(test_images_and_features):
            print(f'Test Image {i} ...', end='')
            # Processed image is the image which has been drawn on to indicate bounding boxes and classes
            predicted_features, processed_image, run_time = feature_detection_marker(
                sift,
                bf,
                gray_query_image,
                colour_query_image,
                training_images_and_paths,
                params,
                show_output=show_output
            )

            if show_output:
                for text, pos, scale, txt_col in image_info:
                    draw_text(
                        processed_image,
                        text=text,
                        pos=pos,
                        font_scale=scale,
                        text_color=txt_col,
                        text_color_bg=(0, 0, 0)
                    )
                for name, bb_tl, bb_br in actual_features:
                    cv2.rectangle(processed_image, bb_tl, bb_br, (0, 255, 0), 2)
                    draw_text(
                        processed_image,
                        name,
                        pos=bb_tl,
                        text_color=(0, 255, 0),
                        text_color_bg=(0, 0, 0)
                    )

                cv2.imshow('Test Image {i}', processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            times.append(run_time)
            predicted_feature_names_set = set([f[0] for f in predicted_features])
            actual_feature_names_set = set([f[0] for f in actual_features])

            # Below determines if predictions are correct, false positives, negatives, etc.
            false_pos_res = ''
            false_neg_res = ''
            if actual_feature_names_set == predicted_feature_names_set:
                correct += 1
                print(GREEN, 'Correct!!!', NORMAL)
            elif predicted_feature_names_set != actual_feature_names_set:
                false_pos_res = predicted_feature_names_set.difference(actual_feature_names_set)
                false_neg_res = actual_feature_names_set.difference(predicted_feature_names_set)
                false_pos += len(false_pos_res)
                false_neg += len(false_neg_res)
                print(RED, 'IN-Correct!!!', NORMAL)

            true_pos_res = predicted_feature_names_set.intersection(actual_feature_names_set)
            if len(predicted_feature_names_set) == 0:
                predicted_feature_names_set = 'None'

            print(BLUE, 'Predicted      :', NORMAL, predicted_feature_names_set)
            print(BLUE, 'Actual         :', NORMAL, actual_feature_names_set)
            print(BLUE, 'True Positives :', NORMAL, true_pos_res if len(true_pos_res) > 0 else 'None')
            if len(false_pos_res) == 0:
                print(BLUE, 'False Positives:', NORMAL, 'None')
            else:
                print(RED, 'False Positives:', NORMAL, false_pos_res)
            if len(false_neg_res) == 0:
                print(BLUE, 'False Negatives:', NORMAL, 'None')
            else:
                print(RED, 'False Negatives:', NORMAL, false_neg_res)
            print(f'Time to do feature matching: {round(run_time, 3)}ms')
            print()

        accuracy = correct * 100 / len(list(test_images_and_features))
        total_false_results = false_pos + false_neg
        print(f'Accuracy: {accuracy}%')
        print(f'Total false results: {total_false_results}')
        print(f'Average runtime per image: {round(np.mean(times), 3)}ms')
        print(BLUE, f'Note: Total false results is the total number of false-positives and false-negatives for all test images', NORMAL)
        print(BLUE, f'Note: The average runtime is for the feature matching process and does not include any additional processing done '
            + 'to check accuracy, display the results, etc.', NORMAL)
    except:
        print(RED, 'Unknown error occurred:', NORMAL, traceback.format_exc())
        exit()

# TODO: remove?
def feature_detection_marker(sift, bf, gray_query_image, colour_query_image, all_training_data, params, show_output=False):
    extra_time = 0
    start_time = time.time()
    query_kp, query_desc = sift.detectAndCompute(gray_query_image, None)

    found_features = []
    for feature_image, feature_image_path in all_training_data:
        current_kp, current_desc = sift.detectAndCompute(feature_image, None)

        # TODO: experiment with this being:
        # NOTE: bf crossCheck param can be True of False but if True threshold test might not be necessary?
        # matches = bf.match(query_desc, current_desc)
        # good_matches = [m for m in matches if m.distance < params['someThreshold']]

        # Get the best matches within a threshold distance
        good_matches = []
        for match in bf.knnMatch(query_desc, current_desc, k=2):
            if len(match) == 2:
                m, n = match
                if m.distance < params['ratioThreshold'] * n.distance:
                    good_matches.append(m)

        if len(good_matches) >= params['min_good_matches']:
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
                    draw_text(
                        colour_query_image,
                        text=feature_name,
                        to_centre=True,
                        pos=centroid,
                        text_color=(255, 0, 0),
                        text_color_bg=(0, 0, 0)
                    )
                    extra_time += time.time() - s_extra

                found_features.append(
                    (feature_name, bb_top_left, bb_bottom_right))
    run_time = time.time() - start_time - extra_time

    return found_features, colour_query_image, run_time


def read_test_dataset(dir, file_ext):
    print(f'Reading test dataset: {dir}')

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

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        all_data.append((
            path,
            img,
            all_features
        ))

    return all_data


def read_training_dataset(dir):
    print(f'Reading training dataset: {dir}')
    training_data = []
    for path in os.listdir(dir + 'png/'):
        img = cv2.imread(dir + 'png/' + path, cv2.IMREAD_GRAYSCALE)
        feature_name = feature_name_from_path(path)
        training_data.append((img, feature_name))
    return training_data

# TODO: remove?
def remove_noise_from_image(image, kernel=np.ones((3, 3), np.uint8)):
    _, thresh_img = cv2.threshold(image, 250, 255, cv2.THRESH_TOZERO_INV)
    eroded_img = cv2.erode(thresh_img, kernel, cv2.BORDER_REFLECT)
    mask = np.uint8(eroded_img <= 20) * 255
    result = cv2.bitwise_or(eroded_img, mask)
    return result

# TODO: remove?
def feature_name_from_path(img_path):
    return img_path[img_path.find('-')+1:img_path.find('.png')]

# TODO: remove?
def oriented_bounding_box(points):
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def draw_text(
    img,
    text,
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
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)

    return text_size
