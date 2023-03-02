import cv2


def feature_detection(image_path, image_paths_to_match_against, sift_params):
    original_query_image = cv2.imread(image_path)
    query_image = cv2.cvtColor(original_query_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(**sift_params)
    query_kp, query_desc = sift.detectAndCompute(query_image, None)

    bf = cv2.BFMatcher()

    found_features = []
    for img_path in image_paths_to_match_against:
        current_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        current_kp, current_desc = sift.detectAndCompute(current_image, None)

        matches = sorted(bf.match(query_desc, current_desc), key=lambda x: x.distance)

        # Get the best match and check if it is within a threshold distance
        if matches[0].distance < 50:
            # TODO: Determine bounding box, draw bounding box, add bounding box coords to found_features
            # NOTE: We get key-points, but there can be noise so we need to determine how to use them

            found_features.append([feature_name_from_path(img_path), (), ()])

    # cv2.imshow('img', original_query_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return found_features

def feature_name_from_path(img_path):
    return img_path[img_path.find('-')+1:img_path.find('.png')]
