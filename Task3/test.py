import os

test_dir = 'Task3/Task3Dataset/'

def read_test_dir(test_dir):
    image_files = sorted(os.listdir(test_dir + 'images/'))
    actual_image_features = []
    
    for image_file in image_files:
        print(image_file)
        csv_file = test_dir + 'annotations/' + image_file[:-4] + '.csv'
        with open(csv_file, 'r') as fr:
            features = fr.read().splitlines()
        for feature in features:
            # feature = feature.split(", ")
            end_of_class = feature.find(", ")
            end_of_tuple = feature.find("), (") + 1
            feature_class = feature[:end_of_class]
            print(feature_class)
            feature_coord1 = eval(feature[end_of_class + 2:end_of_tuple])
            print(feature_coord1)
            feature_coord2 = eval(feature[end_of_tuple + 2:])
            print(feature_coord2)

            actual_image_features.append([feature_class, feature_coord1, feature_coord2])

    return image_files, actual_image_features

images, actual_features = read_test_dir(test_dir)

for image, feature in zip(images, actual_features):
    print(image, feature)


