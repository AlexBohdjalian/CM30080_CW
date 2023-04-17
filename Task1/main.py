from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import warnings
import cv2


def get_intersection_point(line1, line2):
    # get the intersection point of two lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    # if lines are not parallel
    if det != 0:
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return [x, y]
    else:
        return None


def get_intersection_point_estimate(lines, img_dim, params_dbscan):
    # find all the intersection points of all the lines
    intersection_points = []

    for line1 in lines:
        for line2 in lines:
            if not np.array_equal(line1, line2):
                intersection_point = get_intersection_point(line1, line2)
                if intersection_point is not None:
                    x, y = intersection_point
                    # ensure intersection point is within image bounds
                    if 0 <= x <= img_dim[1] and 0 <= y <= img_dim[0]:
                        intersection_points.append((x, y))

    if not intersection_points:
        return None

    # use DBSCAN to find the largest cluster of intersection points and ignore noise
    clustering = DBSCAN(**params_dbscan).fit(intersection_points)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    largest_cluster_points = np.array(intersection_points)[labels == largest_cluster_label]
    intersection_point_estimate = np.mean(largest_cluster_points, axis=0)

    return intersection_point_estimate


def get_direction_vector(line, intersection):
    # get the direction vector of a line

    dist = np.linalg.norm(line[0:2] - intersection)

    # orient the direction vector away from the intersection point
    if np.linalg.norm(line[2:4] - intersection) > dist:
        dx, dy = line[2] - line[0], line[3] - line[1]
    else:
        dx, dy = line[0] - line[2], line[1] - line[3]

    # normalise the direction vector
    length = np.sqrt(dx**2 + dy**2)
    unit_vector = [dx / length, dy / length]

    return unit_vector


def determine_angles(input_data, params):
    results = []

    for img_path, actual_angle in input_data:

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_dim = img.shape[:2]

        # get edges using canny-edge detection
        edges = cv2.Canny(img, **params['CANNY'])

        # get lines using Hough transform
        lines = cv2.HoughLinesP(edges, **params['HOUGH'])
        lines = np.squeeze(lines, axis=1)

        # at least two lines are needed to determine the angle
        if len(lines) < 2:
            warnings.warn(f"Less than 2 lines detected for {img_path}")
            results.append((img_path, actual_angle, None))
            continue

        # get the intersection point estimate in order to determine the direction vectors
        intersection_point_estimate = get_intersection_point_estimate(lines, img_dim, params['DBSCAN'])

        if intersection_point_estimate is None:
            warnings.warn(f"Intersection point estimate is None for {img_path}")
            results.append((img_path, actual_angle, None))
            continue

        # get the direction vectors of all the lines using estimated intersection point
        direction_vectors = [get_direction_vector(line, intersection_point_estimate) for line in lines]

        # use KMeans to find the direction vectors of the two lines in the image
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(direction_vectors)
        except ConvergenceWarning:
            warnings.warn(f"ConvergenceWarning for {img_path}")
            results.append((img_path, actual_angle, None))
            continue

        vec1, vec2 = kmeans.cluster_centers_

        # get the angle between the two direction vectors
        dot_product = np.dot(vec1, vec2)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

        if np.isnan(angle_rad):
            warnings.warn(f"Angle is NaN for {img_path}")
            results.append((img_path, actual_angle, None))
            continue

        angle = round(np.rad2deg(angle_rad))

        results.append((img_path, actual_angle, angle))

    return results



