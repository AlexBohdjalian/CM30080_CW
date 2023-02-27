import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import os

RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

def print_correct(text):
    print(GREEN + text + NORMAL)

def print_wrong(text):
    print(RED + text + NORMAL)

def get_vector_from_line(line, intersection):
    dist = np.linalg.norm(line[0:2] - intersection)

    if np.linalg.norm(line[2:4] - intersection) > dist:
        return np.array([line[2] - line[0], line[3] - line[1]])
    
    return np.array([line[0] - line[2], line[1] - line[3]])

def angle_between_lines(line1, line2):
    inter_point = line_intersection(line1, line2)

    vec1 = get_vector_from_line(line1, inter_point)
    vec2 = get_vector_from_line(line2, inter_point)

    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    return np.rad2deg(angle)

def line_intersection(line1, line2):
    # Calculate intersection point of two lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if the two lines are parallel
    if det == 0:
        raise ValueError("Lines are parallel, angle is undefined.")

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    return px, py


def determine_angles(results, show_output=True):
    correct = []
    wrong = []
    for img_path, actual_angle in results:
        # Read image to grayscale
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.COLOR_RGB2GRAY)

        # Apply gaussian blur to improve canny edge detection
        # img = cv2.GaussianBlur(img, (5, 5), 0)

        # _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        # img = cv2.GaussianBlur(img, (3, 3), 0)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        # Detect edges, then edges from lines
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        # cv2.imshow('edges', edges)
        # cv2.waitKey(0)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 29, minLineLength=10, maxLineGap=10)

        try:
            # K-means clustering
            lines = lines.reshape(-1, 4)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(lines)
            labels = kmeans.labels_
        except:
            if show_output:
                print('Error while clustering')

        try:
            line_label_dict = defaultdict(list)
            line_label_dict.update((label, [np.round(line)]) for line, label in zip(lines, labels))

            if len(line_label_dict) < 2:
                raise ValueError('insufficient lines detected')
            line1, line2 = [np.mean(line_label_dict[i], axis=0) for i in range(2)]
        except Exception as e:
            if show_output:
                print('Error while getting lines', e)

        try:
            angle = angle_between_lines(line1, line2)
            angle = round(angle)

            message = 'The angle in {} is\t {:>4} deg   {:>4} deg'.format(img_path, angle, actual_angle)
            if angle == actual_angle:
                if show_output:
                    print_correct(message)
                correct.append(img_path)
            else:
                # if abs(actual_angle - angle) <= 5:
                #     line1 = [round(thing) for thing in line1]
                #     line2 = [round(thing) for thing in line2]
                #     cv2.line(img, (line1[0], line1[1]), (line1[2], line1[3]), (0,255,0), 2)
                #     cv2.line(img, (line2[0], line2[1]), (line2[2], line2[3]), (0,255,0), 2)
                #     cv2.imshow('image', img)
                #     cv2.waitKey(0)
                if show_output:
                    print_wrong(message)
                wrong.append((img_path, actual_angle, angle))
        except:
            if show_output:
                print('Error while calculating angle')

    return correct, wrong
