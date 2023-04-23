import shutil
import cv2
import os
import numpy as np
import time


# NOTE: For marker, we have assumed that the additional data you have is in the same format as the data given.
# Please replace the two directories below with your own and then execute this file.
test_dir = 'Task2/Task2Dataset/TestWithoutRotations'
train_dir = 'Task2/Task2Dataset/Training/png'

trained_filters_location = 'Task2/trained_filters/'

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

    # NORMALISE filter
    #window_mean_difference = filter-filter.mean()
    #filter = window_mean_difference/np.sqrt(np.sum(window_mean_difference**2))

    kernel_position=[0,0] # tracks position of top left corner of kernal/filter

    while (kernel_position[0]+filter.shape[0])<=input_image.shape[0]: #repeatedly move kernel down
        #move kernel to the right
        while (kernel_position[1]+filter.shape[1])<=input_image.shape[1]:
            # get chunk of image at current location of filter
            current_image_chunk = input_image[
                kernel_position[0]:kernel_position[0]+filter.shape[0],
                kernel_position[1]:kernel_position[1]+filter.shape[1]
            ]

            #NORMALIZE IMAGE WINDOW
            #window_mean_difference = current_image_chunk-current_image_chunk.mean()
            #current_image_chunk = window_mean_difference/np.sqrt(np.sum(window_mean_difference**2))

            # calculate SSD between chunk and filter
            chunk_SSD = np.sum((current_image_chunk-filter)**2) # from [255,128,0],[1,239,254] we get []
            #save SSD result to output image at appropriate indices
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

        # --- ROTATE FILTERS AND APPEND -- taken from code for task 3
        # TODO: might need to increase this for better performance
        # TODO: sift should be invariant of image rotation so why does this improve performance?
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
TARGET_PYRAMID_MIN_RES =32
TEST_PYRAMID_MIN_RES = 256

TARGET_SCORE_INTENSITY = 2.8e-06
TARGET_SCORE_STD = 1.18 # 1.01
# target score intensity imagesize_scaler
# target score  std imagesize_scaler

GAUSS_SIZE = 5
GAUSS_SIGMA = 0.25


def test():
    true_positives=0
    false_positives=0
    true_negatives=0
    false_negatives=0
    correct = 0
    incorrect = 0 # if (presence wrong) this is wrong
    # if (presence right and position wrong) this is wrong
    # if (presence wrong OR position wrong) this is wrong


    # retreive trained filters
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


        # -- GET NEXT OBJECT TO CHECK FOR. Search test image for this object using gaussian pyramid progressively
        # load gaussian pyramid for selected object from files
        for folder, target_object_filters in all_folders_and_filters:
            # --- define SEARCH AREA (y1,y2,x1,x2)
            current_testing_square = [
                0,test_image.shape[0],
                0,test_image.shape[1]
            ]

            # ------- START GOING UP THE GAUSSIAN TREE OF THE TEST IMAGE - testing section is modified as we go up
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
                    best_convolved = SSD(test_image_crop, best_filter) / 2000

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
                        convolved = SSD(test_image_crop, target_object_filters[i].copy()) / 2000
                        #print(convolved.max(), convolved.shape)

                        #cv2.imshow('convolved',convolved)
                        #cv2.waitKey(1)
                        range_ = convolved.max()-convolved.min()
                        #conv_top_pixels_cluster = np.where(convolved<=convolved.min()+range_*0.3) # find  lowest 30% pixels for intensity
                        #score_intensity = convolved[conv_top_pixels_cluster].mean()
                        score_intensity = convolved.min()

                        std_conv_top_pixels_cluster = np.where(convolved<=convolved.min()+range_*0.1)  # find  lowest 10% pixels for std

                        std_coords = np.mean([std_conv_top_pixels_cluster[0].std(), std_conv_top_pixels_cluster[1].std()]) 
                        score_std = std_coords
                        # num_cluster_std = std_conv_top_pixels_cluster[0].shape[0]
                        #print('std_cluster',num_cluster_std)

                        if score_intensity<best_intensity_score and (score_std<best_std_score):
                            best_filter_index = i
                            best_intensity_score = score_intensity
                            best_std_score = score_std
                            gauss_p_layer_filter = best_filter_index # store which gaussian pyramid level in the target object best matches the test image at the test iamges smallest gaussian pyramid level
                            best_convolved=convolved.copy()
                            best_filter = target_object_filters[i].copy()

                            ###cv2.imshow('',convolved/convolved.max())
                            ###cv2.waitKey(1)

                # ----- GET OBJECT PRESENCE PREDICTION
                filter_size = target_object_filters[best_filter_index].shape[0]*target_object_filters[best_filter_index].shape[1] ** 2   # compare all intensities to max possible intensity, which is (max_difference=1-0)*(num_filter_elements=hxw)**2
                is_object_in_image=False
                best_intensity_score /= filter_size # compensate for size of filter

                if best_intensity_score<TARGET_SCORE_INTENSITY and best_std_score<TARGET_SCORE_STD:
                    is_object_in_image=True

                # if model says target object is not in image and it is, label asnwer as WRONG - it skips checking the rest of the gauss pyramid in the test iamge at this point so we can assess it immediately without waiting for top pyramid level answer
                if (not is_object_in_image and (folder in label_objects)):
                    false_negatives+=1
                    incorrect+=1
                    ###cv2.imshow('',convolved/convolved.max())
                    ###cv2.waitKey(1)
                    break
                # if model says target object is not in image and it is not, label asnwer as RIGHT - it skips checking the rest of the gauss pyramid in the test iamge at this point so we can assess it immediately without waiting for top pyramid level answer
                elif (not is_object_in_image and not (folder in label_objects)):
                    true_negatives+=1
                    correct+=1

                try:
                    accuracy = correct/(correct+incorrect)
                except:
                    pass

                # ------- IF TARGET OBJECT IN IMAGE, GET POSITION
                if not is_object_in_image:
                    # print(f'{target_object}\t not in image, skipping rest of filters for this objects')
                    break # if no filter from this object is seen at this leve, it wont be seen at any level - skip this object
                else:
                    # --- GET MOST LIKELY POSITION FOR FILTER -> THRESHOLD CONVOLVED IMAGE, CLUSTER -> PREDICT POSITION  @@@@@@@@@@@@@@ TODO: K-MEANS CLSTERING OF HIGHEST INTENSITY POINTS

                    conv_top_pixels_cluster = np.where(best_convolved==best_convolved.min())  # find  lowest STD area and select as position of object
                    mean_x = np.median(conv_top_pixels_cluster[1])
                    mean_y = np.median(conv_top_pixels_cluster[0])

                    # OFFSET PREDICTED COORDS ACCORDING TO IMAGE CROP
                    mean_x += current_testing_square[2]
                    mean_y += current_testing_square[0]
                    mean_x += best_filter.shape[1]//2
                    mean_y += best_filter.shape[0]//2 #compensate for padding during SSD
                    mean_x=round(mean_x)
                    mean_y=round(mean_y)

                    # UPDATE IMAGE CROP - new area is 10% larger than selected square
                    current_testing_square[0] = max(0,                    int(mean_y - (best_filter.shape[0]/2)*1.1))
                    current_testing_square[1] = min(test_image.shape[0],  int(mean_y + (best_filter.shape[0]/2)*1.1))
                    current_testing_square[2] = max(0,                    int(mean_x - (best_filter.shape[1]/2)*1.1))
                    current_testing_square[3] = min(test_image.shape[1],  int(mean_x + (best_filter.shape[1]/2)*1.1))

                    # --- FINAL DECISION IS ONLY MADE FOR HIGHEST RES TEST IMAGE - DONT ASSESS ACCURACY BEFORE FINAL DECISION IS MADE
                    if gauss_p_layer_testimage == 0:

                        # --- FROM ESTIMATES CENTRE POSITION, GET BOUDNING BOX, TEXT & DISPLAY --- taken from code in task 3
                        text=folder
                        font=cv2.FONT_HERSHEY_PLAIN
                        font_scale=1
                        font_thickness=1
                        text_color=(0, 0, 0)
                        text_color_bg=(0, 255, 0)
                        text_w, text_h = best_filter.shape[1], best_filter.shape[0]
                        x1 = mean_x - text_w//2
                        y1 = mean_y - text_h//2
                        x2 = mean_x + text_w//2
                        y2 = mean_y + text_h//2
                        out_img = cv2.rectangle(out_img, (x1, y1), (x2, y2), text_color_bg, 2)
                        out_img = cv2.putText(out_img, text, (x1, y1 + text_h+5 + font_scale - 1), font, font_scale, text_color, font_thickness)

                        # check for true positive, get estimated position
                        if folder in label_objects:
                            true_positives+=1
                            idx = label_objects.index(folder)

                            label_pos = np.array(label_positions[idx]).reshape(-1, 2)
                            pred_pos = np.array([mean_x, mean_y]).reshape(-1, 2)
                            position_MSE = np.mean((label_pos - pred_pos)**2)

                            if position_MSE >1:
                                incorrect+=1
                            else:                                
                                correct+=1
                        else:
                            incorrect+=1
                            false_positives+=1

                    current_testing_square[0] = int(current_testing_square[0]*2) # we are going to go up a level on  the gaussian pyramid of the test image, so we need to compensate for this by halving the window size and coords, too
                    current_testing_square[1] = int(current_testing_square[1]*2)
                    current_testing_square[2] = int(current_testing_square[2]*2)
                    current_testing_square[3] = int(current_testing_square[3]*2)

            if folder in label_objects:
                if is_object_in_image:
                    print(f'{GREEN}Feature detected: {folder}{NORMAL}')
                else:
                    print(f'{RED}Feature not detected: {folder}{NORMAL}')
            elif is_object_in_image:
                print(f'{RED}Feature detected: {folder}{NORMAL}')

        # show final image after we have checked for all objects in this test image
        #cv2.imshow('',out_img)
        #cv2.waitKey(1000)
        #cv2.imwrite('predicted_'+test_file, out_img)
    print()
    print('Accuracy       : ' + accuracy)
    print('Total Images   : ' + (correct+incorrect))
    print('False Negatives: ' + false_negatives)
    print('False Positives: ' + false_positives)
    print('True Positives : ' + true_positives)
    print('True Negatives : ' + true_negatives)


start = time.time()
train()
end1 = time.time()
test()
end2 = time.time()

print('Time to train: ' + end1-start)
print('Time to test : ' + end2-start)
