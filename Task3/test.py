import os

test_dir = 'Task3/Task3Dataset/'

def read_test_dir(test_dir):
    image_files = os.listdir(test_dir + 'images/')
    image_files = sorted(image_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

    actual_image_features = []
    
    for image_file in image_files:
        csv_file = test_dir + 'annotations/' + image_file[:-4] + '.csv'
        with open(csv_file, 'r') as fr:
            features = fr.read().splitlines()
        for feature in features:
            end_of_class = feature.find(", ")
            end_of_tuple = feature.find("), (") + 1
            feature_class = feature[:end_of_class]
            feature_coord1 = eval(feature[end_of_class + 2:end_of_tuple])
            feature_coord2 = eval(feature[end_of_tuple + 2:])

            actual_image_features.append([feature_class, feature_coord1, feature_coord2])

    return image_files, actual_image_features

images, actual_features = read_test_dir(test_dir)

for image, feature in zip(images, actual_features):
    print(image, feature)


