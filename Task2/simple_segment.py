import cv2
import os
import numpy as np
import math
import random

test_folder = '/media/idmi/DISSERTATN/CComputer Vision/Task2Dataset/TestWithoutRotations/images/'
test_labels = '/media/idmi/DISSERTATN/CComputer Vision/Task2Dataset/TestWithoutRotations/annotations/'
train_folder='/media/idmi/DISSERTATN/CComputer Vision/Task2Dataset/Training/png/'

# get object names
object_names = os.listdir(train_folder)
for i in range(len(object_names)):
    name = object_names[i].split('.')[0]
    print(name)
    object_names[i] = name.split('-')[1]
print(object_names)

# load train filenames
train_files = os.listdir(train_folder)

# load test filenames
test_files = os.listdir(test_folder)




# ----------------- HYPERPARAMETERS
MIN_ACCEPT_INTENSITY = 0.6
MAX_ACCEPT_STD = 0.9





def MSE(x1, x2):
    return (x1-x2)**2

true_positives=0
false_positives=0
true_negatives=0
false_negatives=0
for test_file in test_files:
    print(test_file)
    #if test_file != 'test_image_20.png':
    #    print('skipping')
    #    continue

    # import test image  -REMOVE BACKGROUND AND SCALE TEST IMAGE
    test_image = cv2.imread(test_folder+test_file)
    test_image_original = test_image.copy()
    test_image[np.where(( test_image > [240,240,240] ).all(axis=2))] = [0,0,0]
    kernel = np.ones((2, 2), np.uint8)
    test_image = cv2.erode(test_image, kernel, cv2.BORDER_REFLECT) 

    test_image = test_image.astype(np.float32)/255.
    for c in range(3):
        test_image[:,:,c] -= test_image[:,:,c].mean()
    

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


    # --- for test image, check all target objects
    for target_object in train_files:
        print('--------------------',target_object)
        #if target_object != '002-bike.png':
        #    print('skipping')
        #    continue
        
        # --- LOAD FILES import train files
        train_image = cv2.imread(train_folder+target_object)

        # --- REMOVE BACKGROUND FROM TRAINING
        train_image[np.where(( train_image > [240,240,240] ).all(axis=2))] = [0,0,0]
        kernel = np.ones((2, 2), np.uint8)
        train_image = cv2.erode(train_image , kernel, cv2.BORDER_REFLECT) 

        #test_image  = np.clip(test_image,a_min=0.0, a_max=255.0)
        #test_image = test_image.astype(np.uint8)





        # --- CREATE GAUSSIAN TREE FROM TRAIN IMAGE
        filters = []
        filters.append(cv2.GaussianBlur(train_image, (5,5), 0))
        for c in range(0):
            filters[0][:,:,c] = filters[0][:,:,c] / filters[0][:,:,c].sum()
            #filters[0][:,:,c] = filters[0][:,:,c] - filters[0][:,:,c].mean()
        loop=True
        i=0
        while loop:
            filters.append( cv2.GaussianBlur(filters[i], (5,5), 0) )# apply gaussian blur
            res = np.asarray([filters[i].shape[0], filters[i].shape[1]])
            filters[i+1] = cv2.resize(filters[i], res//2, interpolation=cv2.INTER_AREA) # downscale
            #cv2.imshow('filter',filters[i+1])
            #cv2.waitKey(100)
                
            i+=1
            if ((res//2)[0]==64) or ((res//2)[1]==64):
                loop=False
                
        # scale filters
        for i in range(len(filters)):
            filter = filters[i].astype(np.float32)/255.
            for c in range(3):
                filter[:,:,c] = filter[:,:,c] / filter[:,:,c].sum()
                filter[:,:,c] = filter[:,:,c] - filter[:,:,c].mean()
            filters[i]=filter
            
        # --- ROTATE FILTERS AND APPEND
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
                rotated_filter = cv2.warpAffine(filter, rotated_filter, (bound_w, bound_h), borderValue=(0,0,0))
                rotated_filters.append(rotated_filter)
        filters = filters+rotated_filters







        # -- CONVOLVE ALL TARGET OBJECT ACROSS IMAGE, FIND BEST MATCH & GET POSITION
        print('searching for all levels and rotations...')
        best_match = np.zeros((filters[0].shape[0], filters[0].shape[1]))
        best_match_intensity=0
        best_filter_index=0
        std_coords=0
        for i in range(len(filters)):
            #print(i,end='')
            # --- CONVOLVE FILTERS ACROSS TEST IMAGE
            # check position detection of every filter    
            convolved = test_image.copy()
            filter = filters[i]
            for c in range(3):
                convolved[:,:,c] = cv2.filter2D(src=test_image[:,:,c], ddepth=-1, kernel=filter[:,:,c])
            convolved = convolved.sum(axis=2)
            #cv2.imshow('convolved',test_image)
            #cv2.waitKey(100)
            #cv2.imshow('convolved',convolved)
            #cv2.waitKey(500)

            # --- CHECK IF CURRENT CONVOLVED IMAGE HAS HIGHER INTENSITY MATCH THAN PREVIOUS BEST. update previous best if so
            #get intensity of best match so far:
            best_top_pixels_intensity = best_match.max()-(best_match.max()-best_match.min())/30.
            best_top_pixels_cluster = np.where(best_match>=best_top_pixels_intensity)
            best_match_intensity = best_match[best_top_pixels_cluster].mean()
            #get intwensity for current convolution
            conv_top_pixels_intensity = convolved.max()-(convolved.max()-convolved.min())/30.
            conv_top_pixels_cluster = np.where(convolved>=conv_top_pixels_intensity)
            convolved_intensity = convolved[conv_top_pixels_cluster].mean()
            # select image with highest intensity as best match
            #print('convolved intenity:',convolved_intensity)
            if convolved_intensity > best_match_intensity:
                #print('new best match!', convolved_intensity)
                best_match = convolved
                best_match_intensity = convolved_intensity
                best_filter_index=i
                std_coords = np.mean([conv_top_pixels_cluster[0].std(), conv_top_pixels_cluster[1].std()]) 

            
        print('intense, std=', best_match_intensity, std_coords)
        # GET OBJECT PRESENCE PREDICTION
        # get if estimator guessed object presence correctly or not
        is_object_in_image=False
        if best_match_intensity>MIN_ACCEPT_INTENSITY and std_coords<MAX_ACCEPT_STD: #---------------------------------------------------------------- O
            is_object_in_image=True
        predicted_object = target_object.split('.')[0].split('-')[-1]
        print('presence guess: = ',is_object_in_image,predicted_object)
        print('presence label: = ', (predicted_object in label_objects) )

        if (is_object_in_image and (predicted_object in label_objects)):
            true_positives+=1
            print('intensity',best_match_intensity)
        elif (is_object_in_image and not (predicted_object in label_objects)):
            false_positives+=1
            print('intensity',best_match_intensity)
        elif (not is_object_in_image and (predicted_object in label_objects)):
            false_negatives+=1
            print('intensity',best_match_intensity)
        elif (not is_object_in_image and not (predicted_object in label_objects)):
            true_negatives+=1
        accuracy = (true_positives+true_negatives) / (true_positives+true_negatives+false_negatives+false_positives)
        print(accuracy)


        # If target object found in image, do proper processing
        if is_object_in_image:


            # --- GET MOST LIKELY POSITION FOR FILTER -> THRESHOLD CONVOLVED IMAGE, CLUSTER -> PREDICT POSITION  @@@@@@@@@@@@@@ TODO: K-MEANS CLSTERING OF HIGHEST INTENSITY POINTS
            top_pixels_intensity = best_match.max() -(best_match.max()-best_match.min())/10.   # GET TOP K HIGHEST INTENSITY PIXELS
            top_pixels_cluster = np.where(best_match>=top_pixels_intensity) 
            mean_x = round(top_pixels_cluster[1].mean())
            mean_y = round(top_pixels_cluster[0].mean())
            std_coords = np.mean([top_pixels_cluster[0].std(), top_pixels_cluster[1].std()]) 


            # --- FROM ESTIMATES CENTRE POSITION, GET BOUDNING BOX, TEXT & DISPLAY
            img=test_image_original.copy()
            text=target_object
            font=cv2.FONT_HERSHEY_PLAIN
            font_scale=1
            font_thickness=1
            text_color=(10, 10, 10)
            text_color_bg=(0, 255, 0)
            x, y = int(mean_x), int(mean_y)
            text_w, text_h = filters[best_filter_index].shape[1], filters[best_filter_index].shape[0]
            x -= text_w // 2
            y -= text_h // 2
            x1 = x + text_w
            y1 = y + text_h
            out_img = cv2.rectangle(img, (x, y), (x1, y1), text_color_bg, 2)
            boundboxed = cv2.putText(out_img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)




            # ---- OUTPUT ESTIMATIONS, GET ERROR
            # if object is present, get position error
        
            print("FOUND MATCH: SHOWING | position",mean_x,mean_y)
            print('coords std:', std_coords)
            cv2.imshow('estimated',boundboxed)
            cv2.imshow('convoled',best_match)
            cv2.waitKey(0)

            if predicted_object in label_objects:
                idx = label_objects.index(predicted_object)
                print('predicted position', x,y,x1,y1)
                print('label position', label_positions[idx])

                print(' ---------- position error MSE = ', np.mean([MSE(label_positions[idx][0], x), 
                                                                    MSE(label_positions[idx][1], y), 
                                                                    MSE(label_positions[idx][2], x1), 
                                                                    MSE(label_positions[idx][3], y1)
                                                                    ]))