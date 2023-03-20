
import cv2
import os
import numpy as np
import math
import random
import os
    
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






def test():


    true_positives=0
    false_positives=0
    true_negatives=0
    false_negatives=0

    correct = 0
    incorrect = 0 # if (presence wrong) this is wrong
    # if (presence right and position wrong) this is wrong
    # if (presence wrong OR position wrong) this is wrong



    # rereeive trained filters
    trained_filters_dirs = os.listdir('trained_filters/')

    # load test image
    for test_file in test_files:
        print(test_file)
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

        # import test image  -REMOVE BACKGROUND AND SCALE TEST IMAGE
        test_image = cv2.imread(test_folder+test_file)
        test_image_original = test_image.copy()
        test_image[np.where(( test_image > [240,240,240] ).all(axis=2))] = [0,0,0]
        kernel = np.ones((2, 2), np.uint8)
        test_image = cv2.erode(test_image, kernel, cv2.BORDER_REFLECT) 

        test_image = test_image.astype(np.float32)/255.
        for c in range(3):
            test_image[:,:,c] -= test_image[:,:,c].mean()
        
        cv2.imshow('test_image',test_image)
        cv2.waitKey(1)

        for folder in trained_filters_dirs:
            target_object = '-'.join(folder.split('-')[1:])

            filters = []
            filters_files = os.listdir('trained_filters/'+folder)
            for filter_file in filters_files:
                # load filter from file (255)
                filter = cv2.imread('trained_filters/'+folder+'/'+filter_file)

                # scale filter (0-1, sum to one, remove mean)
                filter = filter.astype(np.float32)/255.
                for c in range(3):
                    filter[:,:,c] = filter[:,:,c] / filter[:,:,c].sum()
                    filter[:,:,c] = filter[:,:,c] - filter[:,:,c].mean()
                filters.append(filter)

            



            # -- FOR TARGET OBJECT, CONVOLVE ALL its filters ACROSS IMAGE.
            # Then, FIND BEST MATCHING FILTER.
            # If any of thsi filters make a good match, the target object is in the image.
            print('searching for all levels and rotations...', target_object, label_objects)
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
                
                #cv2.imshow('convolved',convolved)
                #cv2.waitKey(1)

                # --- CHECK IF CURRENT CONVOLVED IMAGE HAS HIGHER INTENSITY MATCH THAN PREVIOUS BEST. update previous best if so
                #get intensity of best match so far:
                best_top_pixels_intensity = best_match.max()*0.7 #-(best_match.max()-best_match.min())/30.
                best_top_pixels_cluster = np.where(best_match>=best_top_pixels_intensity)
                best_match_intensity = best_match[best_top_pixels_cluster].mean()
                #get intwensity for current convolution
                conv_top_pixels_intensity = convolved.max()*0.7 #-(convolved.max()-convolved.min())/30.
                conv_top_pixels_cluster = np.where(convolved>=conv_top_pixels_intensity)
                convolved_intensity = convolved[conv_top_pixels_cluster].mean()
                # select image with highest intensity as best match
                std_conv_top_pixels_intensity = convolved.max()*0.9
                std_conv_top_pixels_cluster = np.where(convolved>=std_conv_top_pixels_intensity)
                std_convolved_intensity = convolved[std_conv_top_pixels_cluster].mean()
                #print('convolved intenity:',convolved_intensity)
                if convolved_intensity > best_match_intensity:
                    #print('new best match!', convolved_intensity)
                    best_match = convolved
                    best_match_intensity = convolved_intensity
                    best_filter_index=i
                    std_coords = np.mean([std_conv_top_pixels_cluster[0].std(), std_conv_top_pixels_cluster[1].std()]) 
                    background_std = convolved.std()
                cv2.imshow('bestmatch',best_match)
                cv2.waitKey(1)
                
            


            
            # GET OBJECT PRESENCE PREDICTION
            # get if estimator guessed object presence correctly or not
            print('intense, std=', best_match_intensity, std_coords)
            is_object_in_image=False

            if False:
                target_intensity = MIN_ACCEPT_INTENSITY
                target_std = MAX_ACCEPT_STD
                if filters[best_filter_index].shape[0] < 64:
                    target_intensity = 0.8
                    target_std = 0.7        
                if (best_match_intensity>target_intensity and std_coords<target_std): #---------------------------------------------------------------- O
                    is_object_in_image=True
            if True:
                if (best_match_intensity>0.49 and std_coords<1.2 and background_std<0.071) or (background_std<0.055 and best_match_intensity>0.71) or (background_std<0.03 and std_coords<0.95 and best_match_intensity>0.26) or(background_std<0.09 and std_coords<0.9 and best_match_intensity>0.9): #---------------------------------------------------------------- O
                    is_object_in_image=True



            print(test_file,'presence guess: = ',is_object_in_image,target_object)
            print(test_file, 'presence label: = ', (target_object in label_objects) )

            if (is_object_in_image and (target_object in label_objects)):
                true_positives+=1
                correct+=1
                print('intensity',best_match_intensity)
            elif (is_object_in_image and not (target_object in label_objects)):
                false_positives+=1
                incorrect+=1
                print('!!!WRONG: FLASE POSITIVE intensity, std, backstd',best_match_intensity, std_coords, background_std,filters[best_filter_index].shape[0])
                cv2.imshow('bestmatch',best_match)
                input('')
            elif (not is_object_in_image and (target_object in label_objects)):
                false_negatives+=1
                incorrect+=1
                print('!!! WRONG: FALSE NEGATIVE intensity, std',best_match_intensity, std_coords, background_std,filters[best_filter_index].shape[0])
                cv2.imshow('bestmatch',best_match)
                input('')
            elif (not is_object_in_image and not (target_object in label_objects)):
                true_negatives+=1
                correct+=1
            
            #accuracy = (true_positives+true_negatives) / (true_positives+true_negatives+false_negatives+false_positives)
            #print(accuracy)
            accuracy = correct/(correct+incorrect)
            print('ACC----------------------------',accuracy)

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
                cv2.imshow('bestmatch',best_match)
                cv2.waitKey(1)

                if target_object in label_objects:
                    idx = label_objects.index(target_object)
                    print('predicted position', x,y,x1,y1)
                    print('label position', label_positions[idx])

                    position_MSE = np.mean([MSE(label_positions[idx][0], x), 
                                                                        MSE(label_positions[idx][1], y), 
                                                                        MSE(label_positions[idx][2], x1), 
                                                                        MSE(label_positions[idx][3], y1)
                                                                        ])

                    print(' ---------- position error MSE = ', position_MSE)
                    if position_MSE >1:
                        print("WRONG POSITION")
                        input('')
                    
test()
