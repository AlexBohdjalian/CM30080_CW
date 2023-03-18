# CM30080_CW
Computer Vision Group CW

### Task1: Measuring the angle
TODO: Describe to examiner how the code works and how to use it on their own data

### Task2: 
TODO: Describe to examiner how the code works and how to use it on their own data

### Task3: SIFT Feature Matching
For this task, we have implemented the SIFT algorithm for feature matching with the given datasets. There are four main files:

1. `main.py`. This file contains the core functionality for feature matching using the SIFT algorithm. It includes functions for reading the datasets, image pre-processing, SIFT feature extraction, feature matching, transformation estimation, and result visualisation. To use it on your own data, we recommend you use the `marker_test.py` file, which has the best feature matching parameters we could generate and only requires you to indicate the directories for your data.

2. `marker_test.py`. This file is specifically designed for marker use or debugging. It contains the necessary code to test the SIFT feature matching on a given dataset. To use this file, replace the three directory paths near the end of the file with the paths to your datasets. Then, simply run the script to see how our code performs. For each image the script will output the Predicted results and Actual results, indicating if the algorithm was successful. Finally, it will give the Accuracy, False-Positives, True-Positives **(NOT YET)**, False-Negatives, average runtime, and other relevant information.

3. `hyperopt_parameter_tuning.py`. This file is used for tuning the parameters of the SIFT algorithm and the matching process. It employs a hyperparameter optimisation library to find the optimal values for parameters such as the ratio test threshold and the RANSAC threshold. Please note that running this file may take a considerable amount of time (1.5+ hours) depending on your machine, and the parameter space.

4. `graph_production.py`. This file contains the code used to produce the graphs for our performance analysis. It **...** and then generates graphs to visualise the results.

We found that the processing time for an average test image using our tuned parameters is about 4.5ms **(VERIFY)**. This demonstrates the effectiveness of our SIFT implementation and its suitability for feature matching tasks in various computer vision applications.
