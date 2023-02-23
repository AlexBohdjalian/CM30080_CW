import cv2
import numpy as np
from sklearn.cluster import KMeans

RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

def print_correct(text):
    print(GREEN + text + NORMAL)

def print_wrong(text):
    print(RED + text + NORMAL)

def get_vector_from_intersection(line, intersection):
    dist = np.linalg.norm(line[0:2] - intersection)

    if np.linalg.norm(line[2:4] - intersection) > dist:
        return np.array([line[2] - intersection[0], line[3] - intersection[1]])
    
    return np.array([line[0] - intersection[0], line[1] - intersection[1]])

def angle_between_lines(line1, line2):
    inter_point = line_intersection(line1, line2)

    vec1 = get_vector_from_intersection(line1, inter_point)
    vec2 = get_vector_from_intersection(line2, inter_point)

    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    return np.rad2deg(angle)

def line_intersection(line1, line2):
    # Calculate intersection point of two lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    return px, py



with open('Task1/assets/list.txt', 'r') as f:
    results = f.readlines()
results = [(item.split(',')[0], int(item.split(',')[1].strip())) for item in results]

for file_name, actual_angle in results:
    img_path = 'Task1/assets/' + str(file_name)
    img = cv2.imread(img_path, cv2.COLOR_RGB2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=250)

    try:
        # K-means clustering
        lines = lines.reshape(-1, 4)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(lines)
        labels = kmeans.labels_
        line_label_dict = {}

        for line, label in zip(lines, labels):
            line_label_dict[label] = line_label_dict.get(label, []) + [np.round(line)]

        line1 = np.mean(line_label_dict[0], axis=0)
        line2 = np.mean(line_label_dict[1], axis=0)

        angle = angle_between_lines(line1, line2)
        angle = round(angle)

        message = f'The angle in {file_name} is \t{angle} deg \t{GREEN}{actual_angle} deg{NORMAL}'
        if angle == actual_angle:
            print_correct(message)
        else:
            print_wrong(message)
    except:
        print('Error occurred')
