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
        for line in lines:
            cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0))
        cv2.imshow('img', img)
        cv2.waitKey(2000)

        # K-means clustering
        lines = lines.reshape(-1, 4)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(lines)
        labels = kmeans.labels_
        line_label_dict = {}

        line1 = None
        line2 = None
        for line, label in zip(lines, labels):
            line_label_dict[label] = line_label_dict.get(label, []) + [np.round(line)]

        line1 = np.mean(line_label_dict[0], axis=0)
        line2 = np.mean(line_label_dict[1], axis=0)

        print(line1, line2)

        delta_x = line2[2] - line2[0]
        delta_y = line2[3] - line2[1]
        angle = np.arctan2(delta_y, delta_x) - np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
        angle = round(abs(angle * 180 / np.pi))

        message = f'The angle in {file_name} is \t{angle} deg \t{GREEN}{actual_angle} deg{NORMAL}'
        if angle == actual_angle:
            print_correct(message)
        else:
            print_wrong(message)
    except:
        print('Error occured')
