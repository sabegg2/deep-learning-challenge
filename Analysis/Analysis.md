# deep-learning-challenge ANALYSIS
- Module 21 Challenge
- Steph Abegg

# Report on the Neural Network Model

Deep Learning Charity Funding Outcome Predictor Project using hyper-tuned Neural Networks.

## Overview:

I've created a tool for the nonprofit foundation Alphabet Soup that can help it select applicants for funding with the best chance of success in their ventures. Using my knowledge of machine learning and neural networks, I have used the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. We were set a target of 75% accuracy for our model. From Alphabet Soup’s business team, I received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

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

- EIN and NAME—Identification columns removed from the input data because they are neither targets nor features 

- Created cutoff point to bin "rare" categorical variables together in a new value, Other for both CLASSIFICATION and APPLICATION_TYPE 

- Converted categorical data to numeric with pd.get_dummies, split the preprocessed data into features and target arrays, then lastly split into training and tesing datasets 

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
 
#### 2: Compiling, Training, and Evaluating the Model

I build the base model with the following parameters with low computation time in mind:

- Two hidden layers with 80, 30 neurons split. The hidden layer activation function was set to relu.

- Output node is 1 neuron as it was binary classifier model with only one output: was the funding application succesful, yes or no? The output layer activation function was set to sigmoid as the model output is binary classification between 0 and 1.

The model prediction gave an Accuracy: 0.7294.
  
(Note that I also tried other techniques for the base model, such as: I increased the hidden layers to 3 and set the third hidden layer at 30. I also tried using the tanh activation and 3 hidden layers with 90, 30, 30 neurons split and a sigmoid activation for output. I also experimented with increasing nodes and neurons. But despite doing this all models came below the 75% threshold.)

### 3: Optimize the Model

I decided to use an automated model optimizer to get the most accurate model possible by creating method that creates a keras Sequential model using the keras-tuner library with hyperparametes options. I created four optimized models in an attempt to get the accuracy to at least 75%.

#### Optimized Model V1

Here are the changes I made from the base model:

- Five hidden layers with a number of neurons between 1 and 80 (first layer) and 1 and 40 (other layers) and activation function choice of either relu or tanh.
- max_epochs=20
- 60 trials

The model prediction gave an Accuracy: 0.7298

#### Optimized Model V2

Here are the changes I made from the base model:

- Five hidden layers with a number of neurons between 1 and 80 (first layer) and 1 and 40 (other layers) and activation function choice of either relu or tanh.
- max_epochs=20
- Added a learning rate choice: Allow Keras Tuner to explore different learning rates. The learning rate is a critical hyperparameter that can significantly impact the training of the model.
- Dropout Layers: Include dropout layers to reduce overfitting by randomly setting a fraction of input units to 0 at each update during training.
- Regularization: Add L2 regularization to the dense layers to help prevent overfitting.
- 60 trials

The model prediction gave an Accuracy: 0.7262

#### Optimized Model V3

Here are the changes I made from the base model:

- Eight hidden layers with a number of neurons between 1 and 100 (first layer) and 1 and 50 (other layers) and activation function choice of either relu or tanh.
- max_epochs=60. The number of epochs refers to how many times the learning algorithm will work through the entire training dataset. It is a crucial hyperparameter that can significantly affect the performance and behavior of a machine learning model. Here’s how the number of epochs impacts the model. Low Number of Epochs: Can lead to underfitting and high bias. High Number of Epochs: Can result in overfitting and high variance. Choosing the right number of epochs is a balancing act, and it's often beneficial to experiment with different values to find the one that yields the best model performance on your specific dataset.
- Batch Size=50 (the default--used in previous Optimized Models--is 32). This means that the training data is be divided into batches of 50 samples each during training. The choice of batch size can significantly affect the performance and convergence of your model. Smaller batch sizes might lead to more noisy updates, while larger batch sizes might provide a more stable estimate of the gradient, but they may require more memory.
- Early Stopping: Implemented as a callback during the tuning process to stop training when the validation loss does not improve. This is useful if the number of epochs is high.
- 177 trials

The model prediction gave an Accuracy: 0.7269

#### Optimized Model V4

Here are the changes I made from the base model:

- Five hidden layers with a number of neurons between 1 and 80 and activation function choice of either relu or tanh.
- max_epochs=20
- Removed 'ASK_AMT' (which was predominatly $5000 and a then just one or two occurences of all other values) column, and added back in the 'NAME' column.
- Creating more bins for rare occurrences in columns (specifically, created two Other bins Other1 and Other2 for the CLASSIFICATION column).
- Decreased the number of values in the Other bin for APPLICATION_TYPE (speficically, set it to v<156 rather than v<528).
- 60 trials
  
The model prediction gave an Accuracy: 0.7289

## Final Optimization

The initial optimization model used five hidden layers with a number of neurons between 1 and 80 (first layer) and 1 and 40 (other layers) and activation function choice of either relu or tanh, and 20 epochs.

I made various attempts to better the accuracy of the initial optimised model: adding a learning rate choice, including dropout layers to reduce overfitting, adding L2 regularization to the dense layers to help prevent overfitting, adding more neurons to a hidden layer, adding more hidden layers, using different activation functions for the hidden layers, adding more epochs to the training regimen, adding early stopping to stop training when the validation loss does not improve. This is useful if the number of epochs is high, dropping more or fewer columns, creating more bins for rare occurrences in columns, increasing or decreasing the number of values for each bin.

Interestingly, most of these modifications did not give improvement over the initial optimization model. The one modification that did result in increasing the accuracy above 75% was adding the Name column back into the model. A possible explaination is that this reduces some of the "noise" in the oversampled data and allows the algorithm to further classify the data.

## Summary:

The final automatically optimized neural network trained model from the keras tuner method achieved 80% prediction accuracy with a 0.45 loss, using a sigmoid activation function with input node of 76; 5 hidden layers at a 16, 21, 26, 11, 21, neurons split and 50 training epochs. Performing better than the non automized model. Keeping the Name column was crucial in achieving and and going beyond the target. This shows the importance of the shape of your datasets before you preprocess it. The explanation being that it reduces some of the "noise" in the oversampled data and allows the algorithm to further classify the data.
