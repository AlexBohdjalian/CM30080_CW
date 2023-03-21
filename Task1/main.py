from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
import numpy as np
import warnings
import cv2

RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'


def print_correct(text):
    print(GREEN + text + NORMAL)


def print_wrong(text):
    print(RED + text + NORMAL)


def average_intersection_point(lines1, lines2):

    # Compute all intersection points between both clusters lines
    intersection_points = []
    for line1 in lines1:
        for line2 in lines2:
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if det != 0:
                x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / det
                y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / det
                intersection_points.append([x, y])

    # Compute the average intersection point
    if intersection_points:
        return np.mean(intersection_points, axis=0)
    else:
        return None


def get_vector_from_line(line, intersection):
    # Get vector from furthest point on line to intersection point

    dist = np.linalg.norm(line[0:2] - intersection)

    if np.linalg.norm(line[2:4] - intersection) > dist:
        return np.array([line[2] - line[0], line[3] - line[1]])

    return np.array([line[0] - line[2], line[1] - line[3]])


def get_lines_gradients(lines):
    # Find gradient of lines in degrees to handle vertical lines
    gradients = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            gradient = np.pi / 2
        else:
            gradient = np.arctan((y2-y1)/(x2-x1))

        gradients.append([gradient])

    return gradients


def determine_angles(results):
    correct = []
    wrong = []
    for img_path, actual_angle in results:

        img = cv2.imread(img_path, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(img, 78, 281, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 0.9426758401015772, 0.0008438020371663023, 40, minLineLength=41.91973871274722, maxLineGap=34.14817528989028)

        if len(lines) < 2:
            warnings.warn(f"Less than 2 lines detected for {img_path}")
            wrong.append(img_path)
            continue

        # Get gradients of all lines found in degrees
        gradients = get_lines_gradients(lines)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                # Cluster by gradient
                kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(gradients)
        except ConvergenceWarning:
            wrong.append(img_path)
            continue

        # Split lines into clusters
        cluster1, cluster2 = [], []
        for i, line in enumerate(lines):
            if kmeans.labels_[i] == 0:
                cluster1.append(line[0])
            else:
                cluster2.append(line[0])

        # Get the average intersection of every line in cluster1 with every line in cluster2
        avg_intersection = average_intersection_point(cluster1, cluster2)

        # Get the average vectors of each cluster
        vec1 = np.mean([get_vector_from_line(line, avg_intersection) for line in cluster1], axis=0)
        vec2 = np.mean([get_vector_from_line(line, avg_intersection) for line in cluster2], axis=0)

        # Get the angle between the two vectors
        angle_rad = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        if np.isnan(angle_rad):
            wrong.append(img_path)
            continue

        angle = round(np.rad2deg(angle_rad))

        if angle == actual_angle:
            correct.append(img_path)
        else:
            wrong.append(img_path)

    return correct, wrong



