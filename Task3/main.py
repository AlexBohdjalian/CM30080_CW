import cv2
import numpy as np
cv2.setRNGSeed(0)


def feature_detection_optimiser(image_path, images_to_match_against, all_sifts, bf):
    query_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    results = []
    for i in range(len(all_sifts)):
        print(i)
        _, query_desc = all_sifts[i].detectAndCompute(query_image, None)

        found_features = []
        for current_name, current_image in images_to_match_against:

            _, current_desc = all_sifts[i].detectAndCompute(current_image, None)

            matches = sorted(bf.match(query_desc, current_desc), key=lambda x: x.distance)

            # Get the best match and check if it is within a threshold distance
            if matches[0].distance < 50:
                found_features.append([feature_name_from_path(current_name), (), ()])
        results.append(found_features)
    return results

def feature_detection(image_path, image_paths_to_match_against, params, show_output=False):
    query_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # TODO: check that gaussian blur is being applied because query_image contains specs of blurry dots (for artificial noise)

    sift = cv2.SIFT_create(
        nfeatures=params['sift']['nfeatures'],
        nOctaveLayers=params['sift']['nOctaveLayers'],
        contrastThreshold=params['sift']['contrastThreshold'],
        edgeThreshold=params['sift']['edgeThreshold'],
        sigma=params['sift']['sigma']
    )
    query_kp, query_desc = sift.detectAndCompute(query_image, None)
    bf = cv2.BFMatcher(
        normType=params['BFMatcher']['normType'],
        crossCheck=params['BFMatcher']['crossCheck']
    )

    feature_keypoints = {}
    for img_path in image_paths_to_match_against:
        current_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        current_kp, current_desc = sift.detectAndCompute(current_image, None)
        matches = bf.match(query_desc, current_desc)

        # Get the best matches within a threshold distance
        good_matches = [m for m in matches if m.distance < params['matchThreshold']]
        if len(good_matches) > 0:
            feature_name = feature_name_from_path(img_path)
            matched_query_kp = [query_kp[m.queryIdx] for m in good_matches]
            matched_query_xy = [(int(kp.pt[0]), int(kp.pt[1])) for kp in matched_query_kp]

            # Store matched keypoints in dictionary
            if feature_name not in feature_keypoints:
                feature_keypoints[feature_name] = []
            feature_keypoints[feature_name].extend(matched_query_xy)

    found_features = []
    for feature_name, keypoints in feature_keypoints.items():
        # Compute the bounding box of the matched keypoints
        bb_1 = (
            min([xy[0] for xy in keypoints]),
            min([xy[1] for xy in keypoints])
        )
        bb_2 = (
            max([xy[0] for xy in keypoints]),
            max([xy[1] for xy in keypoints])
        )

        # if show_output:
        #     cv2.rectangle(orig_query_image, bb_1, bb_2, (0, 255, 0), 2)
        #     draw_text(orig_query_image, text=feature_name, to_centre=True, pos=(bb_1[0], bb_1[1] - 5))

        found_features.append([feature_name, bb_1, bb_2])

    # if show_output:
    #     cv2.imshow("query_image", orig_query_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return found_features

def feature_name_from_path(img_path):
    return img_path[img_path.find('-')+1:img_path.find('.png')]

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