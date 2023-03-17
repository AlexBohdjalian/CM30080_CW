import cv2
import numpy as np

cv2.setRNGSeed(0)


def feature_detection_hyperopt(sift, bf, query_image, all_training_data_kp_desc, params):
    query_kp, query_desc = sift.detectAndCompute(query_image, None)

    found_features = []
    for feature_path, current_kp, current_desc in all_training_data_kp_desc:
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
            M, mask = cv2.findHomography(
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
                feature_name = feature_name_from_path(feature_path)
                found_features.append((feature_name, (), ()))

    return found_features

def feature_detection_marker(query_image, image_paths_to_match_against, params, show_output=False):
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

    found_features = []
    for current_image, current_image_path in image_paths_to_match_against:
        current_kp, current_desc = sift.detectAndCompute(current_image, None)

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
            M, mask = cv2.findHomography(
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
                feature_name = feature_name_from_path(current_image_path)
                matched_query_kp = [query_kp[m.queryIdx] for m in good_matches]
                matched_query_xy = [(int(kp.pt[0]), int(kp.pt[1])) for kp in matched_query_kp]

                box = oriented_bounding_box(np.array(matched_query_xy))
                bb_top_left = np.min(box, axis=0)
                bb_bottom_right = np.max(box, axis=0)

                if show_output:
                    cv2.drawContours(query_image, [box], 0, (0, 255, 0), 2)
                    draw_text(query_image, text=feature_name, to_centre=True, pos=(bb_top_left[0], bb_top_left[1] - 5))

                found_features.append((feature_name, bb_top_left, bb_bottom_right))

    if show_output:
        cv2.imshow("query_image", query_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    return found_features

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