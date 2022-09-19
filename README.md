# Digit-Recognizer
This project ranks amongst my very first educational projects in which i implemented a shallow neural network with 1 hidden layer for predicting the digit shown in the images in Octave.

All the algorithms such as Forward Propagation, Back Propagation, etc is implemented from the scratch in different functions below:

  - Main_Program.m
  - displayData.m
  - sigmoid.m
  - random_initialization.m
  - Forward_Propagation.m
  - cost_function.m
  - back_propagation.m
  - fmincg.m
  - predict.m

The dataset consists of 5000 (20, 20, 1) pictures in Input.mat file.

The weights consist of 2 matrices namely, Theta1 and Theta2 with 25 * 401 and 10 * 26 dimensions respectively.

The hidden layer and output layer consist of 25 and 10 neurons respectively.

fmincg.m is an advanced optimization algorithm provided by Coursera.com.
