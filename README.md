# deep-learning-challenge REPORT
# Report on the Neural Network Model
- Module 21 Challenge
- Steph Abegg

## Refresher on Neural Network Models:

Neural network models are a class of machine learning algorithms inspired by the structure and function of the human brain. They are composed of layers of interconnected units called neurons (also known as nodes), which process information and learn to make predictions or decisions based on the input data.

Key Components of Neural Networks:

- Neurons (Nodes): The basic unit of a neural network. Each neuron takes in multiple inputs, processes them through a weighted sum, and applies an activation function to produce an output.

- Layers: Neural networks are organized in layers:
  - Input layer: Receives the raw data (e.g., images, text, numerical data).
  - Hidden layers: Intermediate layers between the input and output layers. These layers perform complex transformations on the   - input data through learned weights.
  - Output layer: Produces the final predictions (e.g., classification labels, regression values).

- Weights and Biases: Weights are the parameters that the model learns during training, which determine the importance of each input to a neuron. Biases are additional parameters added to the weighted sum to allow the model more flexibility.

- Activation Functions: Activation functions introduce non-linearity to the network, enabling it to learn complex patterns. Common activation functions include ReLU, sigmoid, and tanh.

## Overview of the Challenge:

The full instructions for this challenge are in [INSTRUCTIONS.md](INSTRUCTIONS.md).

In this module challenge, we were tasked with creating a tool for the nonprofit foundation Alphabet Soup that can help the foundation select applicants for funding with the best chance of success in their ventures. Using machine learning and neural networks, we used the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. The set target was 75% model accuracy. Alphabet Soup’s business team provided a CSV [charity_data.csv](Resources/charity_data.csv) containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- `EIN` and `NAME` — Identification columns
- `APPLICATION_TYPE` — Alphabet Soup application type
- `AFFILIATION` — Affiliated sector of industry
- `CLASSIFICATION` — Government organization classification
- `USE_CASE` — Use case for funding
- `ORGANIZATION` — Organization type
- `STATUS` — Active status
- `INCOME_AMT` — Income classification
- `SPECIAL_CONSIDERATIONS` — Special considerations for application
- `ASK_AMT` — Funding amount requested
- `IS_SUCCESSFUL` — Was the money used effectively

## Steps Taken:

### 1: Data Preprocessing

- Dataset was checked for null and duplicated values

- `EIN` and `NAME` columns removed from the input data because they are neither targets nor features ('EIN' is a unique reference number given to each organization requesting/receiving funding, and `NAME` is the name of the organization making a request/receiving funding from the foundation).

- Cutoff points were created to bin "rare" categorical variables together in a new value, `Other`, for both `CLASSIFICATION` and `APPLICATION_TYPE`. (`CLASSIFICATION` - Due to the number of unique values within this column, the values have been 'binned' into an Other category if the value is < 1883. `APPLICATION_TYPE` - Due to the number of unique values within this column, the values have been 'binned' into an Other category if the value is < 528.)

- Categorical data was conveted into numeric with pd.get_dummies.

- The preprocessed data was split into features and target arrays.
  
  - Target Variable for the model:

      - `IS_SUCCESSFUL`
      
  - Feature Variables for the model:

      - `APPLICATION_TYPE`
      - `AFFILIATION`
      - `CLASSIFICATION`
      - `USE_CASE`
      - `ORGANIZATION`
      - `STATUS`
      - `INCOME_AMT`
      - `SPECIAL_CONSIDERATIONS`
      - `ASK_AMT`
   
- The preprocessed was split into training and testing datasets.
 
### 2: Compiling, Training, and Evaluating the Model

[AlphabetSoupCharity_InitialModel.ipynb](Notebooks/AlphabetSoupCharity_InitialModel.ipynb)

[AlphabetSoupCharity.keras](Models_keras_files/AlphabetSoupCharity.keras)

The base base neural network model was built using the `tenserflow` library `keras` package  with the following parameters with low computation time in mind:

- Two hidden layers with 80, 30 neurons split. The hidden layer activation function was set to relu. (ReLU--Rectified Linear Unit--is generally preferred for deep neural networks due to faster convergence and fewer issues with vanishing gradients, although it can suffer from the "dying ReLU" problem, where neurons can stop learning if they output zero consistently. The other popular activation choices is tanh--hyperbolic tangent--which can be useful in specific contexts where the model benefits from output values that are symmetric around zero, though it's more prone to vanishing gradients.)

- Output node is 1 neuron as it was binary classifier model with only one output: was the funding application successful, yes or no? The output layer activation function was set to sigmoid as the model output is binary classification between 0 and 1.

The model prediction gave an Accuracy: 0.7292 and Loss: 0.5591.
  
(Note that other techniques were also tried for the base model, such as: increasing the hidden layers to 3 and setting the third hidden layer at 30 neurons; using the tanh activation, experimenting with different numbers of neurons in the hidden layers. But despite doing this all models came below the 75% accuracy threshold.)

Model Summary and loss and accuracy plots from model training:

<img src="Images\parameters_initialmodel.png" width=500>
<img src="Images\plot_loss.png" width=400> <img src="Images\plot_accuracy.png" width=400>


### 3: Optimize the Model

The goal was to get the accuracy of the model to at least 75%.

Attempts to optimize the model made use of the `keras_tuner` library. This provides the ability to test a number of different options for the model, including:

- The number of different hidden layers within the model

- How many neurons per layer

- Varying activation functions

- Ultimatley outputting the parameters which worked best for model accuracy

#### Optimized Model V1

[AlphabetSoupCharity_Optimization_V1.ipynb](Notebooks/AlphabetSoupCharity_Optimization_V1.ipynb)

[AlphabetSoupCharity_Optimization_V1.h5](Models_h5_and_keras_files/AlphabetSoupCharity_Optimization_V1.h5)

The first optimization run using the keras_tuner library had the following options:
- 1-5 Hidden Layers
- Activation functions either:
    - relu
    - tanh
- Up to 80 nodes in the input layer, and up to 40 nodes in the hidden layers
- 20 epochs
  
The best model when ran with 60 trials produced:
- Accuracy: 0.7328
- Loss: 0.5545

Model Summary:

<img src="Images\hyperparameters_model1.png" width=200>

#### Optimized Model V2

[AlphabetSoupCharity_Optimization_V2.ipynb](Notebooks/AlphabetSoupCharity_Optimization_V2.ipynb)

[AlphabetSoupCharity_Optimization_V2.h5](Models_h5_and_keras_files/AlphabetSoupCharity_Optimization_V2.h5)

The second optimization run using the keras_tuner library had the following options:
- 1-5 Hidden Layers
- Activation functions either:
    - relu
    - tanh
- Up to 80 nodes in the input layer, and up to 40 nodes in the hidden layers
- 20 epochs
- Added a learning rate choice, which allows Keras Tuner to explore different learning rates. The learning rate is a critical hyperparameter that can significantly impact the training of the model.
- Include dropout layers to reduce overfitting by randomly setting a fraction of input units to 0 at each update during training.
- Added L2 regularization to the dense layers to help prevent overfitting.
  
The best model when ran with 60 trials produced:
- Accuracy: 0.7317
- Loss: 0.5632

Model Summary:

<img src="Images\hyperparameters_model2.png" width=200>

#### Optimized Model V3

[AlphabetSoupCharity_Optimization_V3.ipynb](Notebooks/AlphabetSoupCharity_Optimization_V3.ipynb)

[AlphabetSoupCharity_Optimization_V3.h5](Models_h5_and_keras_files/AlphabetSoupCharity_Optimization_V3.h5)

The third optimization run using the keras_tuner library had the following options:
- 1-8 Hidden Layers
- Activation functions either:
    - relu
    - tanh
- Up to 100 nodes in the input layer, and up to 50 nodes in the hidden layers
- 60 epochs. The number of epochs refers to how many times the learning algorithm will work through the entire training dataset. It is a crucial hyperparameter that can significantly affect the performance and behavior of a machine learning model. A low number of epochs can lead to underfitting and high bias. A high number of epochs can result in overfitting and high variance. Choosing the right number of epochs is a balancing act, and it's often beneficial to experiment with different values to find the one that yields the best model performance on your specific dataset.
- Batch size of 50 (the default--used in previous optimized models--is 32). This means that the training data is be divided into batches of 50 samples each during training. The choice of batch size can significantly affect the performance and convergence of your model. Smaller batch sizes might lead to more noisy updates, while larger batch sizes might provide a more stable estimate of the gradient, but they may require more memory.
- Implemented Early Stopping as a callback during the tuning process to stop training when the validation loss does not improve. This is useful if the number of epochs is high.
  
The best model when ran with 177 trials produced:
- Accuracy: 0.7336 
- Loss: 0.5541

Model Summary:

<img src="Images\hyperparameters_model3.png" width=200>

#### Optimized Model V4

[AlphabetSoupCharity_Optimization_V4.ipynb](Notebooks/AlphabetSoupCharity_Optimization_V4.ipynb)

[AlphabetSoupCharity_Optimization_V4.h5](Models_h5_and_keras_files/AlphabetSoupCharity_Optimization_V4.h5)

The fourth optimization run using the keras_tuner library had the following options:
- 1-5 Hidden Layers
- Activation functions either:
    - relu
    - tanh
- Up to 100 nodes in the input layer, and up to 50 nodes in the hidden layers
- 20 epochs
- Added back in the 'NAME' column.
  
The best model when ran with 60 trials produced:
- Accuracy: 0.7964
- Loss: 0.4569

Model Summary:

<img src="Images\hyperparameters_model4.png" width=200>

#### Optimized Model V5

[AlphabetSoupCharity_Optimization_V5.ipynb](Notebooks/AlphabetSoupCharity_Optimization_V5.ipynb)

[AlphabetSoupCharity_Optimization_V5.h5](Models_h5_and_keras_files/AlphabetSoupCharity_Optimization_V5.h5)

The fifth optimization run using the keras_tuner library had the following options:
- 1-5 Hidden Layers
- Activation functions either:
    - relu
    - tanh
- Up to 100 nodes in the input layer, and up to 50 nodes in the hidden layers
- 20 epochs
- Removed 'ASK_AMT' (which was predominantly $5000 and a then just one or two occurrences of all other values) column.
- Added back in the 'NAME' column.
- Created more bins for rare occurrences in columns (specifically, created two Other bins Other1 and Other2 for the CLASSIFICATION column).
- Decreased the number of values in the Other bin for APPLICATION_TYPE (specifically, set it to v<156 rather than v<528).
  
The best model when ran with 60 trials produced:
- Accuracy: 0.7967
- Loss: 0.4706

Model Summary:

<img src="Images\hyperparameters_model5.png" width=200>

## Summary

The initial optimization model used five hidden layers with a number of neurons between 1 and 80 (first layer) and 1 and 40 (other layers) and activation function choice of either relu or tanh, and 20 epochs. The model prediction gave an accuracy of 72.9% and loss of 55.9%. Accuracy measures the proportion of correct predictions made by the model out of all predictions. Loss is a measure of how far the predicted values are from the true values, as well as the confidence of the prediction; loss is essentially a penalty for incorrect predictions.

The goal was to get the model accuracy to at least 75%. At the same time, it would be ideal to lower the model loss.

Various attempts were made to better the accuracy of the initial optimized model. These attempts included combinations of the following: adding a learning rate choice, including dropout layers to reduce overfitting, adding L2 regularization to the dense layers to help prevent overfitting, adding more neurons to a hidden layer, adding more hidden layers, using different activation functions for the hidden layers, adding more epochs to the training regimen, adding early stopping to stop training when the validation loss does not improve, dropping more (namely `ASK_AMT`) or fewer (namely `NAME`) columns, creating more bins for rare occurrences in columns, increasing or decreasing the number of values for each bin.

Interestingly, most of these modifications did not give improvement over the initial optimization model. The one modification that did result in increasing the accuracy above 75% was adding the `NAME` column back into the model. After adding the `NAME` column back into the model, the automatically optimized neural network trained model from the keras tuner method achieved 79.7% prediction accuracy and 47.1% loss. This model used a tanh activation function with input node of 11 neurons and 5 hidden layers at a 6, 36, 31, 36, 21 neurons split and 20 training epochs, and a sigmoid output activation function. This model performed better than the non-automized model. 

This likely indicates that the name feature is providing valuable information to the model, either directly (like reflecting demographics) or indirectly (as a proxy for other relevant factors). The names in this dataset are things like "Blue Knights Motorcycle Club", "Genetic Research Institute of the Desert", or "Joseph E Peebles Foundation". While there are thousands of different names, there are some repeated words, such as "Club", "Institute", or "Foundation" that might carry some information that the machine learning model can use. This shows the imporatnce of not discounting the possible information carried by any variable.

