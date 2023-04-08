import shutil
import cv2
import os
import numpy as np
import math
import random
import os
import time



test_folder = 'Task2Dataset/TestWithoutRotations/images/'
test_labels = 'Task2Dataset/TestWithoutRotations/annotations/'
train_folder='Task2Dataset/Training/png/'
trained_filters_location = 'trained_filters/'





# get object names
object_names = os.listdir(train_folder)
for i in range(len(object_names)):
    name = object_names[i].split('.')[0]
    object_names[i] = name.split('-')[1]

# load train filenames
train_files = os.listdir(train_folder)

# load test filenames
test_files = os.listdir(test_folder)




def MSE(x1, x2):
    return (x1-x2)**2


def gaussian_pyramid(image, min_side_length):
     # --- CREATE GAUSSIAN TREE FROM TRAIN IMAGE
    pyramid = []
    # append original image +gaussian to filters
    pyramid.append(cv2.GaussianBlur(image, (5,5), 0))
    loop=True
    i=0
    while loop:

        
        # --- scale down filter size by half compared to previous filter
        new_res = np.asarray([pyramid[-1].shape[0], pyramid[-1].shape[1]]) // 2
        
        new_filter = cv2.resize(pyramid[-1], new_res, interpolation=cv2.INTER_AREA) # downscale

        # --- apply gaussian blur to newest filter size
        new_filter = cv2.GaussianBlur(new_filter, (GAUSS_SIZE,GAUSS_SIZE), GAUSS_SIGMA)


        #cv2.imshow('filter',filters[i+1])
        #cv2.waitKey(100)
        pyramid.append(new_filter)

        if (new_res[0]==min_side_length) or (new_res[1]==min_side_length): # dont make filters smaller than key amount (once ssmall enough, the iamge cant be idenitifed at that small a scale and the filters are easily wrongly triggered)
            break

        i+=1
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
            current_image_chunk = input_image[kernel_position[0]:kernel_position[0]+filter.shape[0],
                                              kernel_position[1]:kernel_position[1]+filter.shape[1]]
            
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

    shutil.rmtree(trained_filters_location)
    os.mkdir(trained_filters_location)

    # --- for test image, check all target objects
    for target_object in train_files:
        #if target_object != '002-bike.png':
        #    print('skipping')
        #    continue
        
        # --- LOAD FILES import train files
        train_image = cv2.imread(train_folder+target_object)

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

        object_folder = trained_filters_location+target_object.split('.')[0]+'/'
        os.mkdir(object_folder)
        for i in range(len(filters)):
            filter=filters[i]
            print(filter)
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
# meaure complexity space, time






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
    trained_filters_dirs = os.listdir(trained_filters_location)

    # load test image
    for test_file in test_files:

        
        #if test_file != 'test_image_20.png':
        #    print('skipping')
        #    continue

    

        # --- load test labels
        test_label_file = test_labels + test_file.split('.')[0] + '.txt'
        label_objects=[]
        label_positions=[]
        with open(test_label_file,'r') as file:
            test_label_file = file.readlines()
        print(test_label_file)
        for line in test_label_file:
            splitline = line.split(',')
            object = splitline[0]
            label_objects.append(object)

            positions = [   int(splitline[1].replace('(','')), int(splitline[2].replace(')','')), int(splitline[3].replace('(','')), int(splitline[4].replace(')',''))   ]
            label_positions.append(positions)

        # ----- import test image  
        # REMOVE BACKGROUND AND SCALE TEST IMAGE
        test_image = cv2.imread(test_folder+test_file)
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
        for folder in trained_filters_dirs:
            target_object = '-'.join(folder.split('-')[1:])
            filters_files = sorted(os.listdir('trained_filters/'+folder))
            


            # get all filters in pyramid for target object
            target_object_filters=[]
            for filter_file in filters_files:
                # load filter from file (255)
                filter_ = cv2.imread('trained_filters/'+folder+'/'+filter_file)

                # scale filter (0-1, sum to one, remove mean)
                filter_ = filter_.astype(np.float32)/255.
                #for c in range(3):
                #    filter[:,:,c] = filter[:,:,c] / filter[:,:,c].sum()
                #    filter[:,:,c] = filter[:,:,c] - filter[:,:,c].mean()
                target_object_filters.append(filter_)


            # --- define SEARCH AREA
            current_testing_square = [  0,test_image.shape[0],
                                        0,test_image.shape[1] ] # y1,y2,x1,x2
            
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
                        num_cluster_std = std_conv_top_pixels_cluster[0].shape[0]
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
                if (not is_object_in_image and (target_object in label_objects)):
                    false_negatives+=1
                    incorrect+=1
                    ###cv2.imshow('',convolved/convolved.max())
                    ###cv2.waitKey(1)
                    break
                # if model says target object is not in image and it is not, label asnwer as RIGHT - it skips checking the rest of the gauss pyramid in the test iamge at this point so we can assess it immediately without waiting for top pyramid level answer
                elif (not is_object_in_image and not (target_object in label_objects)):
                    true_negatives+=1
                    correct+=1

                try:
                    accuracy = correct/(correct+incorrect)
                except:
                    pass







                # ------- IF TARGET OBJECT IN IMAGE, GET POSITION
                if not is_object_in_image:
                    print('object not in image, skipping rest of filters for this objects')
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
                    mean_y += best_filter.shape[0]//2 #copmensate for padding during SSD
                    mean_x=round(mean_x)
                    mean_y=round(mean_y)

                    # UPDATE IMAGE CROP - new area is 10% larger than selected square
                    current_testing_square[0] = max(0,                    int(mean_y - (best_filter.shape[0]/2)*1.1))
                    current_testing_square[1] = min(test_image.shape[0],  int(mean_y + (best_filter.shape[0]/2)*1.1))
                    current_testing_square[2] = max(0,                    int(mean_x - (best_filter.shape[1]/2)*1.1))
                    current_testing_square[3] = min(test_image.shape[1],  int(mean_x + (best_filter.shape[1]/2)*1.1))


                    

                    
                    



                    # ---- OUTPUT ESTIMATIONS, GET ERROR
                    # if object is present, get position error


                    # --- FINAL DECISION IS ONLY MADE FOR HIGHEST RES TEST IMAGE - DONT ASSESS ACCURACY BEFORE FINAL DECISION IS MADE
                    if gauss_p_layer_testimage == 0:

                        # --- FROM ESTIMATES CENTRE POSITION, GET BOUDNING BOX, TEXT & DISPLAY --- taken from code in task 3
                        text=target_object
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
                        if target_object in label_objects:
                            true_positives+=1
                            idx = label_objects.index(target_object)

                            position_MSE = np.mean([MSE(label_positions[idx][0], x1), 
                                                    MSE(label_positions[idx][1], y1), 
                                                    MSE(label_positions[idx][2], x2), 
                                                    MSE(label_positions[idx][3], y2)])

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


        # show final image after we have checked for all objects in this test image
        #cv2.imshow('',out_img)
        #cv2.waitKey(1000)
        #cv2.imwrite('predicted_'+test_file, out_img)
    print('ACC----------------------------',accuracy, 'N=',correct+incorrect, false_negatives, false_positives, true_positives, true_negatives)


    
    
    
    
start = time.time()
train()
end1 = time.time()


test()
end2 = time.time()



print('time to train', end1-start, 'time to test', end2-start)
