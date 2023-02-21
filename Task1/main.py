import cv2
import numpy as np

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
    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(img, 100, 255, 2)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 85)

    try:
        # Draw the detected lines for visual aid
        # for line in lines:
        #     for rho,theta in line:
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a*rho
        #         y0 = b*rho
        #         x1 = int(x0 + 3000*(-b))
        #         y1 = int(y0 + 3000*(a))
        #         x2 = int(x0 - 3000*(-b))
        #         y2 = int(y0 - 3000*(a))
        #         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        # cv2.imshow(file_name, img)
        # cv2.waitKey(1)
        line1 = lines[0][0]
        line2 = lines[1][0]

        # For HoughLines
        angle = np.arctan2(np.sin(line2[1]-line1[1]), np.cos(line2[1]-line1[1]))
        angle = round(abs(angle * 180 / np.pi))

        # For HoughLinesP
        # delta_x = line2[2] - line2[0]
        # delta_y = line2[3] - line2[1]
        # angle = np.arctan2(delta_y, delta_x) - np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
        # angle = round(abs(angle * 180 / np.pi))

        message = f'The angle in {file_name} is \t{angle} deg \t{GREEN}{actual_angle} deg{NORMAL}'
        if angle == actual_angle:
            print_correct(message)
        else:
            print_wrong(message)
            # print('Lines...')
            # print(lines)
    except:
        print('Error occured')
