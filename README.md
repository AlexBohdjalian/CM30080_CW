# CM30080_CW
Computer Vision Group CW

## Task1: Measuring the angle
1. `main.py`. This file contains the core functionality to calculate the angle between two lines in the image. The `determine_angles()` function drives the process, however it is recommended to use `marker_test.py` file to run it with the best parameters.
2. `generate_data.py`. This file contains the functionality to generate images of two intersecting lines with random starting points, thickness, and angles.
3. `marker_test.py`. Run this file to evaluate the performance of 'main.py'. The parameters are hard coded in, one just needs to set the `directory` variable to the directory containing the images you wish to test. 

## Task2: Intensity-based filter matching
We implement Sum Squared Difference for intensity based filter matching, and improve efficiency using gaussian pyramids for the test images and train images, progressively going up the pyramid and cropping the search area to be around the corresponding most likely candidate location for the query object in the previous, lower resolution pyramid layer.

Hyper-parameters were searched for manually.

`main.py` is an all-in-one file. 
At the top of the code are lines defining: 
  - where test images are located
  - where test annotations are located
  - where training images are located
  - where the training image gaussian pyramids should be located 

It creates predicated images with bounding boxes and captions in the base directory.

The `train()` function takes in the location of training images (as a global variable, it is not passed) and creates a gaussian pyramid for each image and saves them to appropriate sub-directories.

The `test()` function reads these files and the test images and annotations and runs our SSD-gaussian pyramid intensity based filter matching algorithm and checks the accuracy while saving the predicated images as with bounding boxes and object labels.


## Task3: SIFT Feature Matching
For this task, we have implemented the SIFT algorithm for feature matching with the given datasets. There are four main files:

1. `main.py`. This file contains the core functionality for feature matching using the SIFT algorithm. It includes functions for reading the datasets, image pre-processing, SIFT feature extraction, feature matching, transformation estimation, and result visualisation. To use it on your own data, we recommend you use the `marker_test.py` file, which has the best feature matching parameters we could generate and only requires you to indicate the directories for your data.

2. `marker_test.py`. This file is specifically designed for marker use or debugging. It contains the necessary code to test the SIFT feature matching on a given dataset. To use this file, replace the three directory paths near the end of the file with the paths to your datasets. Then, simply run the script to see how our code performs. For each image it will give: Accuracy, True-Positives, False-Positives and False-Negatives. Finally, it will give a summary of the overall performance.

3. `hyperopt_parameter_tuning.py`. This file is used for tuning the parameters of the SIFT algorithm and the matching process. It employs a hyperparameter optimisation library to find the optimal values for parameters such as the ratio test threshold and the RANSAC threshold. Please note that running this file may take a considerable amount of time (1.5+ hours) depending on your machine, the parameter space and the number of trials.

4. `graph_production.py`. This file contains the code used to produce the graphs for our performance and parameter analysis.
