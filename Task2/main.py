# some imports


test_folder = 'Task2Dataset/TestWithoutRotations/images/'
test_labels = 'Task2Dataset/TestWithoutRotations/annotations/'
train_folder='Task2Dataset/Training/png/'

# get object names
object_names = os.listdir(train_folder)
for i in range(len(object_names)):
    name = object_names[i].split('.')[0]
    print(name)
    object_names[i] = name.split('-')[1]
print(object_names)

# load train filenames
train_files = os.listdir(train_folder)





def training_process(training_data):
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
        # append original image to 
        #filters.append(cv2.GaussianBlur(train_image, (5,5), 0))
        #for c in range(0):
            #filters[0][:,:,c] = filters[0][:,:,c] / filters[0][:,:,c].sum()
            #filters[0][:,:,c] = filters[0][:,:,c] - filters[0][:,:,c].mean()
        loop=True
        i=0
        while loop:

            # --- apply gaussian blur to newest filter size
            new_filter = cv2.GaussianBlur(train_image, (5,5), 0)
            
            # --- scale down filter size by half # downt make filters smaller than key amount (once ssmall enough, the iamge cant be idenitifed at that small a scale and the filters are easily wrongly triggered)
            scale = (2**(i))
            new_res = np.asarray([train_image.shape[0], train_image.shape[1]]) // scale
            print(new_res)
                
            new_filter = cv2.resize(new_filter, new_res, interpolation=cv2.INTER_AREA) # downscale
            filters.append(new_filter)
            #cv2.imshow('filter',filters[i+1])
            #cv2.waitKey(100)

            if (new_res[0]==32) or (new_res[1]==32):
                break

            i+=1

                
            
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

        for filter in filters:
            object_folder = 'trained_filters/'+target_object.split('.')[0]+'/'
            try:
                os.makedirs(object_folder)
            except:
                pass
            cv2.imwrite(object_folder+str(filters.index(filter))+'.png', filter)
    raise Exception('Function not implemented')

def feature_detection(image_path):
    print('NOT IMPLEMENTED')
    raise Exception('Function not implemented')
