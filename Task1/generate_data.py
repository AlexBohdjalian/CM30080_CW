import cv2
import numpy as np


def generate_data(n_images=30, n_folder='assets/test'):
    # Image dimensions
    height = 480
    width = 854

    # Clear file before writing to it
    open(f'{n_folder}/list.txt', 'w').close()

    for i in range(n_images):
        # Create a black image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Generate two random angles in degrees
        angle1 = np.random.randint(0, 180)
        angle2 = np.random.randint(0, 180)

        length1 = np.random.randint(50, 150)
        length2 = np.random.randint(50, 150)

        # Convert the angles to radians
        theta1 = np.deg2rad(angle1)
        theta2 = np.deg2rad(angle2)

        # Generate the coordinates for the first line
        x1 = np.linspace(0, -length1, 1200)
        y1 = np.tan(theta1) * x1

        # Generate the coordinates for the second line
        x2 = np.linspace(0, length2, 1200)
        y2 = np.tan(theta2) * x2

        # Draw the lines
        cv2.line(img, (int(x1[0] + width / 2), int(y1[0] + height / 2)),
                 (int(x1[-1] + width / 2), int(y1[-1] + height / 2)), (255, 255, 255), 2)
        cv2.line(img, (int(x2[0] + width / 2), int(y2[0] + height / 2)),
                 (int(x2[-1] + int(width / 2)), int(y2[-1] + +height / 2)), (255, 255, 255), 2)

        # Calculate the angle between the lines
        dir1 = np.array([x1[-1] - x1[0], y1[-1] - y1[0]])
        dir2 = np.array([x2[-1] - x2[0], y2[-1] - y2[0]])
        cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
        angle_deg = round(np.rad2deg(np.arccos(cos_angle)))

        # Write the angle to a text file
        with open(f'{n_folder}/list.txt', 'a') as f:
            f.write(f'image{i}.png,{angle_deg}\n')

        cv2.imwrite(f'{n_folder}/image{i}.png', img)


if __name__ == '__main__':
    generate_data()

