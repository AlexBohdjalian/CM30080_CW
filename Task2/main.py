import shutil
import cv2
import os
import numpy as np
import time


# NOTE: For the marker, we have assumed that the additional data you have is in the same format as the data given.
# Please replace the two directories below with your own and then execute this file.
test_dir = 'Task2/Task2Dataset/TestWithoutRotations'
train_dir = 'Task2/Task2Dataset/Training/png'




# NOTE: predicted images are saves to the predicted_dir folder below. 
# Gaussian tree filters are saved to trained_filters_location.
trained_filters_location = 'Task2_trained_filters/'
predicted_dir = 'Task2_predicted_/'





try:
    os.mkdir(trained_filters_location)
except:
    pass

NORMAL = '\u001b[0m'
RED = '\u001b[31m'
GREEN = '\u001b[32m'

def read_training_dataset(dir):
    training_data = []
    for path in os.listdir(dir):
        img = cv2.imread(f'{dir}/{path}')
        feature_name = path[path.find('-')+1:path.find('.png')]
        training_data.append((img, feature_name))
    return training_data

def read_test_dataset(dir, file_ext):
    print(f'Reading test dataset: {dir}')

    image_files = os.listdir(dir + 'images/')
    image_files = sorted(image_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

    all_data = []
    for image_file in image_files:
        csv_file = dir + 'annotations/' + image_file[:-4] + file_ext
        with open(csv_file, 'r') as fr:
            features = fr.read().splitlines()
        all_features = []
        for feature in features:
            end_of_class_name_index = feature.find(", ")
            end_of_first_tuple_index = feature.find("), (") + 1
            feature_class = feature[:end_of_class_name_index]
            feature_coord1 = eval(feature[end_of_class_name_index + 2:end_of_first_tuple_index])
            feature_coord2 = eval(feature[end_of_first_tuple_index + 2:])

            all_features.append([feature_class, feature_coord1, feature_coord2])
        path = dir + 'images/' + image_file

        img = cv2.imread(path)

        all_data.append((
            path,
            img,
            all_features
        ))

    return all_data


train_files = read_training_dataset(train_dir)
test_data = read_test_dataset(test_dir + '/', '.txt')


def MSE(x1, x2):
    return (x1-x2)**2


def gaussian_pyramid(image, min_side_length):
    pyramid = [image]

    while True:
        # Reduce image size using pyrDown
        image = cv2.pyrDown(image)

        # Check if image size is greater than or equal to min_side_length
        if min(image.shape[:2]) >= min_side_length:
            pyramid.append(image)
        else:
            break

    return pyramid


def SSD(input_image_i, filter_i):

    input_image = input_image_i.copy().astype(np.float32)
    filter = filter_i.copy().astype(np.float32)

    output_height = (input_image.shape[0] -  filter.shape[0])+1
    output_width = (input_image.shape[1] -  filter.shape[1])+1
    output_image = np.zeros([output_height,output_width])

    kernel_position=[0,0] # tracks position of top left corner of kernel/filter

    while (kernel_position[0]+filter.shape[0])<=input_image.shape[0]: # move kernel down
        #move kernel right
        while (kernel_position[1]+filter.shape[1])<=input_image.shape[1]:
            # get window of test image at current location of filter
            current_image_chunk = input_image[
                kernel_position[0]:kernel_position[0]+filter.shape[0],
                kernel_position[1]:kernel_position[1]+filter.shape[1]
            ]

            # calculate SSD between window and filter
            chunk_SSD = np.sum((current_image_chunk-filter)**2)
            #save SSD result to output image at appropriate index
            output_image[kernel_position[0], kernel_position[1]] = chunk_SSD

            # move kernel rightward 1
            kernel_position[1] += 1

        # if row ended, move kernel all the way to the left and down one row
        kernel_position[1] = 0
        kernel_position[0] += 1
    return output_image


def train():
    if os.path.exists(trained_filters_location):
        shutil.rmtree(trained_filters_location)
    os.mkdir(trained_filters_location)

    # --- for test image, check all target objects
    for train_image, train_image_name in train_files:
        # --- REMOVE BACKGROUND FROM TRAINING
        train_image[np.where(( train_image > [240,240,240] ).all(axis=2))] = [0,0,0]
        kernel = np.ones((2, 2), np.uint8)
        train_image = cv2.erode(train_image , kernel, cv2.BORDER_REFLECT)

        filters = gaussian_pyramid(train_image, min_side_length=TARGET_PYRAMID_MIN_RES)


        # --- ROTATE FILTERS AND APPEND -- taken from code for task 3   NOTE: set 'directions' > 0 to enable rotated filters. DOES NOT work during test time!
        directions = 0
        angles = [360. / directions * i for i in range(directions)] # up, tr, right, br, down, bl, left, tl
        rotated_filters = []
        for filter in filters:
            for angle in angles:
                height, width = filter.shape[:2] # image shape has 3 dimensions
                image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

                rotated_filter = cv2.getRotationMatrix2D(image_center, angle, 1.)

                # rotation calculates the cos and sin, taking absolutes of those.
                abs_cos = abs(rotated_filter[0,0])
                abs_sin = abs(rotated_filter[0,1])

                # find the new width and height bounds
                bound_w = int(height * abs_sin + width * abs_cos)
                bound_h = int(height * abs_cos + width * abs_sin)

                # subtract old image center (bringing image back to origo) and adding the new image center coordinates
                rotated_filter[0, 2] += bound_w/2 - image_center[0]
                rotated_filter[1, 2] += bound_h/2 - image_center[1]

                # rotate image with the new bounds and translated rotation matrix
                rotated_filter_ = cv2.warpAffine(filter, rotated_filter, (bound_w, bound_h), borderValue=(0,0,0))
                rotated_filters.append(rotated_filter_)

        for rot_filter in rotated_filters:
            filters.append(rot_filter)

        object_folder = trained_filters_location+train_image_name.split('.')[0]+'/'
        os.mkdir(object_folder)
        for i in range(len(filters)):
            filter=filters[i]
            filter_index = str(i)
            cv2.imwrite(object_folder+filter_index+'.png', filter)


# ------- PARAMETERS:
TARGET_PYRAMID_MIN_RES = 16
TEST_PYRAMID_MIN_RES = 128

TARGET_SCORE_INTENSITY = 0.0056
TARGET_SCORE_STD = 1.18
NMS_threshold = 30

GAUSS_SIZE = 5
GAUSS_SIGMA = 0.25


def test():
    if os.path.exists(predicted_dir):
        shutil.rmtree(predicted_dir)
    os.mkdir(predicted_dir)

    true_positives=0
    false_positives=0
    true_negatives=0
    false_negatives=0
    correct = 0
    incorrect = 0 # if (presence wrong) this is wrong
    # if (presence right and position wrong) this is wrong
    # if (presence wrong OR position wrong) this is wrong

    # retrieve trained filters
    all_folders_and_filters = [
        (
            folder,
            [
                cv2.imread(
                    os.path.join(trained_filters_location, folder, filter_file)
                ).astype(np.float32) / 255.0
                for filter_file in sorted(os.listdir(os.path.join(trained_filters_location, folder)))
            ]
        ) for folder in os.listdir(trained_filters_location)]

    # load test image
    for test_image_path, test_image, test_image_features in test_data:
        # --- load test labels
        label_objects=[f[0] for f in test_image_features]
        label_positions=[f[1] for f in test_image_features]

        print(f'Testing: {test_image_path} Actual features: {label_objects}')

        # ----- import test image  
        # REMOVE BACKGROUND AND SCALE TEST IMAGE
        test_image_original = test_image.copy()
        test_image[np.where(( test_image > [240,240,240] ).all(axis=2))] = [0,0,0]
        kernel = np.ones((2, 2), np.uint8)
        test_image = cv2.erode(test_image, kernel, cv2.BORDER_REFLECT)
        test_image = test_image.astype(np.float32)/255.
        #for c in range(3):
        #    test_image[:,:,c] -= test_image[:,:,c].mean()
        # create gaussian pyramid for test image
        test_image_pyramid = gaussian_pyramid(test_image, min_side_length=TEST_PYRAMID_MIN_RES)
        

        out_img=test_image_original.copy()

        #cv2.imshow('',test_image)
        #cv2.waitKey(100)


        # for NMS, we predict positions for all objects, store the names and positions, then do NMS
        predicted_objects = {}     # each elements = 'objectname':(centre_position, SSD score)



        # -- GET NEXT OBJECT TO CHECK FOR. Search test image for this object using gaussian pyramid progressively
        # load gaussian pyramid for selected object from files
        for object_name, target_object_filters in all_folders_and_filters:
            # --- define SEARCH AREA (y1,y2,x1,x2)
            current_testing_square = [
                0,test_image.shape[0],
                0,test_image.shape[1]
            ]

            # ------- START GOING UP THE GAUSSIAN TREE OF THE TEST IMAGE - where the test iamge is cropped to is modified as we go up
            gauss_p_layer_filter = -1
            for gauss_p_layer_testimage in reversed(range(len(test_image_pyramid))):
                test_image = test_image_pyramid[gauss_p_layer_testimage]
                test_image_crop = test_image.copy()
                test_image_crop = test_image_crop[current_testing_square[0]:current_testing_square[1], current_testing_square[2]:current_testing_square[3]] # crop to SEARCH AREA

                # ------ OF ALL THE FILTERS FOR THIS TARGET IMAGE(from its gaussian pyramid), CHECK WHICH ONE IS IN THE CURRENT IMAGE (this is needed the first time where we dont know the scale, after we find it then we know, when we are checking teh enxt image up along the test image pyramid, that the filter we need is jsut teh one up a level on the target obejcts gaussian pyramid than teh one we find here:)
                if gauss_p_layer_filter > -1:
                    gauss_p_layer_filter -= 1 # go up the gaussian pyramid from the last iteration
                    best_filter_index = max(0, gauss_p_layer_filter)
                    best_filter = target_object_filters[best_filter_index].copy()
                    best_convolved = SSD(test_image_crop, best_filter)

                    range_ = convolved.max()-convolved.min()
                    best_intensity_score = convolved.min()
                    std_conv_top_pixels_cluster = np.where(convolved<=convolved.min()+range_*0.1)  # find  lowest 10% pixels for std
                    best_std_score = np.mean([std_conv_top_pixels_cluster[0].std(), std_conv_top_pixels_cluster[1].std()])
                else:
                    # check all filters for object
                    best_intensity_score = float('inf')
                    best_std_score = float('inf')
                    best_filter_index = 0
                    for i in range(len(target_object_filters)): # go from largest to smallest - larger is more discriminative and more likely to know trigger false positive
                        # don check filters which are larger than the test area
                        if target_object_filters[i].shape[0]>=test_image_crop.shape[0] or target_object_filters[i].shape[1]>=test_image_crop.shape[1]:
                            continue

                        # --- CONVOLVE FILTER ACROSS TEST IMAGE
                        convolved = SSD(test_image_crop, target_object_filters[i].copy())

                        score_intensity = convolved.min()

                        range_ = convolved.max()-convolved.min()
                        std_conv_top_pixels_cluster = np.where(convolved<=convolved.min()+range_*0.1)  # find  lowest 10% pixels for std feature
                        std_coords = np.mean([std_conv_top_pixels_cluster[0].std(), std_conv_top_pixels_cluster[1].std()]) 
                        score_std = std_coords

                        if score_intensity<best_intensity_score and (score_std<best_std_score):
                            best_filter_index = i
                            best_intensity_score = score_intensity
                            best_std_score = score_std
                            gauss_p_layer_filter = best_filter_index # store which gaussian pyramid level in the target object best matches the test image at the test iamges smallest gaussian pyramid level
                            best_convolved=convolved.copy()
                            best_filter = target_object_filters[i].copy()

                # ----- GET OBJECT PRESENCE PREDICTION
                filter_size = target_object_filters[best_filter_index].shape[0]*target_object_filters[best_filter_index].shape[1] ** 2   # compare all intensities to max possible intensity, which is (max_difference=1-0)*(num_filter_elements=hxw)**2
                is_object_in_image=False
                best_intensity_score /= filter_size # compensate for size of filter

                if best_intensity_score<TARGET_SCORE_INTENSITY and best_std_score<TARGET_SCORE_STD:
                    is_object_in_image=True


                # ------- IF TARGET OBJECT IN IMAGE, GET POSITION
                if not is_object_in_image:
                    # print(f'{target_object}\t not in image, skipping rest of filters for this objects')
                    # since in a previous layer it might have been detected but now not, we need to remove the position that we saved for this object
                    predicted_objects.pop(object_name, 0)
                    break # if no filter from this object is seen at this leve, it wont be seen at any level - skip this object
                
                else:
                    # --- GET MOST LIKELY POSITION FOR FILTER
                    conv_top_pixels_cluster = np.where(best_convolved==best_convolved.min())  # find pixel with lowest SDD score
                    mean_x = np.median(conv_top_pixels_cluster[1])
                    mean_y = np.median(conv_top_pixels_cluster[0])

                    # OFFSET PREDICTED COORDS ACCORDING TO IMAGE CROP
                    mean_x += current_testing_square[2]
                    mean_y += current_testing_square[0]
                    mean_x += best_filter.shape[1]//2
                    mean_y += best_filter.shape[0]//2 #compensate for padding during SSD
                    mean_x=round(mean_x)
                    mean_y=round(mean_y)

                    size = best_filter.shape[0]

                    predicted_objects[object_name] = (np.asarray([mean_y, mean_x]), best_intensity_score, size)

                    # UPDATE IMAGE CROP - new area is 10% larger than selected square
                    current_testing_square[0] = max(0,                    int(mean_y - (best_filter.shape[0]/2)*1.1))
                    current_testing_square[1] = min(test_image.shape[0],  int(mean_y + (best_filter.shape[0]/2)*1.1))
                    current_testing_square[2] = max(0,                    int(mean_x - (best_filter.shape[1]/2)*1.1))
                    current_testing_square[3] = min(test_image.shape[1],  int(mean_x + (best_filter.shape[1]/2)*1.1))


                    for i in range(4):
                        current_testing_square[i] = int(current_testing_square[i]*2) # we are going to go up a level on  the gaussian pyramid of the test image, so we need to compensate for this by halving the window size and coords, too





        # ------- NON-MAXIMA SUPPRESSION ON PREDICTED OBJECT POSITIONS
        # remmebr objects_predictions hold object names, their estimated coords, and SSD score.

        # save all predicted objects to filtered objects. they will be remove as necessary
        final_output_objects = predicted_objects.copy()        
        
        # for each object in filtered objects, check its neighbours and remove ones which are 1) too close AND 2) have a worse SSD score
        for object_name, (coords, SSD_score, size) in predicted_objects.items():

            # check all potential neighbours of the selected object
            for neighbour_name, (neigbour_coords, neighbour_SSD_score, size) in predicted_objects.items():

                if not neighbour_name==object_name: 
                    # get distance to potential neighbour
                    distance = (sum((coords - neigbour_coords)**2))**0.5
                    
                    # If the neighbour is within the threshold distance to the selected object, remove it from the final output.
                    if distance < NMS_threshold and SSD_score < neighbour_SSD_score:
                        #print('[NMS removed]', neighbour_name)
                        final_output_objects.pop(neighbour_name,0)

        


        for object_name, ([mean_y, mean_x], SSD_score, size) in final_output_objects.items():
            # --- FROM ESTIMATES CENTRE POSITION, GET BOUDNING BOX, TEXT & DISPLAY --- taken from code in task 3
            text=object_name
            font=cv2.FONT_HERSHEY_PLAIN
            font_scale=1
            font_thickness=1
            text_color=(0, 0, 0)
            text_color_bg=(0, 255, 0)
            text_w, text_h = size, size
            x1 = mean_x - text_w//2
            y1 = mean_y - text_h//2
            x2 = mean_x + text_w//2
            y2 = mean_y + text_h//2
            out_img = cv2.rectangle(out_img, (x1, y1), (x2, y2), text_color_bg, 2)
            out_img = cv2.putText(out_img, text, (x1, y1 + text_h+5 + font_scale - 1), font, font_scale, text_color, font_thickness)



        cv2.imwrite(predicted_dir + os.path.basename(test_image_path), out_img)

        # check if the predicted objects are actually in the image or not
        for object_name, ([mean_y, mean_x], score, size) in final_output_objects.items():

            # check for true positive, get estimated position
            if object_name in label_objects:
                print(f'{GREEN}Feature detected: {object_name}{NORMAL}')
                true_positives+=1
                idx = label_objects.index(object_name)

                label_pos = np.array(label_positions[idx]).reshape(-1, 2)
                pred_pos = np.array([mean_x, mean_y]).reshape(-1, 2)
                position_MSE = np.mean((label_pos - pred_pos)**2)
                if position_MSE >1:
                    incorrect+=1
                else:                                
                    correct+=1
            else:
                print(f'{RED}Feature detected: {object_name}{NORMAL}')
                incorrect+=1
                false_positives+=1

        # check if any objects in the test image were missed
        for object_name in label_objects:
            if not object_name in final_output_objects.keys():
                print(f'{RED}Feature not detected: {object_name}{NORMAL}')
                incorrect +=1
                false_negatives+=1
                
        # check true negative the model got right (all selectable objects that both the corrects labels and the predicted objects exclude)
        all_objects = [folder for folder in os.listdir(trained_filters_location)]
        true_negatives += len(list(set(all_objects) - set(final_output_objects.keys()).union(set(label_objects))))
        correct += len(list(set(all_objects) - set(final_output_objects.keys()).union(set(label_objects))))


    print()
    print('Accuracy       :', str((true_positives+true_negatives)/(correct+incorrect)))
    print('Total Images   :', str(correct+incorrect) )
    print('False Negatives:', str(false_negatives))
    print('False Positives:', str(false_positives))
    print('True Positives :', str(true_positives))
    print('True Negatives :', str(true_negatives))


start = time.time()
train()
end1 = time.time()
test()
end2 = time.time()

print('Time to train: ' + str(end1-start))
print('Time to test : ' + str(end2-end1))
print('Total : ' + str(end2-start))
