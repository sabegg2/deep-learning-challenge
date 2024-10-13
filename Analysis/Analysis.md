# Report on the Neural Network Model
- Module 21 Challenge
- Steph Abegg

## Overview:

I've created a tool for the nonprofit foundation Alphabet Soup that can help it select applicants for funding with the best chance of success in their ventures. Using my knowledge of machine learning and neural networks, I have used the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. We were set a target of 75% accuracy for our model. From Alphabet Soup’s business team, I received a CSV [charity_data.csv](../Resources/charity_data.csv) containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

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

- `EIN` and `NAME` columns removed from the input data because they are neither targets nor features ('EIN' is a unique reference number given to each organization requesting/receiving funding, and `NAME` is the name of the organization making a request/receiving funding from the foundation)

- Created cutoff point to bin "rare" categorical variables together in a new value, `Other`, for both `CLASSIFICATION` and `APPLICATION_TYPE`. ('CLASSIFICATION' - Due to the numebr of unique values within this column, the values have been 'binned' into an Other category if the value is < 1883. 'APPLICATION_TYPE' - Due to the number of unique values within this column, the values have been 'binned' into an Other category if the value is < 528.)

- Converted categorical data to numeric with pd.get_dummies.

- Split the preprocessed data into features and target arrays.
  
- Split into training and tesing datasets.

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
 
### 2: Compiling, Training, and Evaluating the Model

[AlphabetSoupCharity_InitialModel.ipynb](../Notebooks/AlphabetSoupCharity_InitialModel.ipynb)
[AlphabetSoupCharity.keras](../Models_keras_files/AlphabetSoupCharity.keras)

I build the base model with the following parameters with low computation time in mind:

- Two hidden layers with 80, 30 neurons split. The hidden layer activation function was set to relu.

- Output node is 1 neuron as it was binary classifier model with only one output: was the funding application succesful, yes or no? The output layer activation function was set to sigmoid as the model output is binary classification between 0 and 1.

The model prediction gave an Accuracy: 0.7284.
  
(Note that I also tried other techniques for the base model, such as: I increased the hidden layers to 3 and set the third hidden layer at 30. I also tried using the tanh activation and 3 hidden layers with 90, 30, 30 neurons split and a sigmoid activation for output. I also experimented with increasing nodes and neurons. But despite doing this all models came below the 75% threshold.)

Model Summary and loss and accuraccy plots from model training:

<img src="..\Images\parameters_initialmodel.png" width=200>
<img src="..\Images\plot_loss.png" width=400> <img src="..\Images\plot_accuracy.png" width=400>


### 3: Optimize the Model

The goal was to get the accuracy of the model to at least 75%.

My attempt to optimize the model made use of the keras_tuner library This provides the ability to test a number of different options for the model, including:

- The number of different hidden layers within the model

- How many neurons per layer

- Varying activation functions

- Ultimatley outputting the parameters which worked best for model accuracy

#### Optimized Model V1

[AlphabetSoupCharity_Optimization_V1.ipynb](../Notebooks/AlphabetSoupCharity_Optimization_V1.ipynb)
[AlphabetSoupCharity_Optimized_V1.keras](../Models_keras_files/AlphabetSoupCharity_Optimized_V1.keras)

The first optimization run using the keras_tuner library had the following options:
- 1-5 Hidden Layers
- Activation functions either:
    - relu
    - tanh
- Up to 80 nodes in the input layer, and up to 40 nodes in the hidden layers
- 20 epochs
  
The best model when ran with 60 trials produced:
- Accuracy: 0.7334
- Loss: 0.5507

Model Summary:

<img src="..\Images\hyperparameters_model1.png" width=200>

#### Optimized Model V2

[AlphabetSoupCharity_Optimization_V2.ipynb](../Notebooks/AlphabetSoupCharity_Optimization_V2.ipynb)
[AlphabetSoupCharity_Optimized_V2.keras](../Models_keras_files/AlphabetSoupCharity_Optimized_V2.keras)

The first optimization run using the keras_tuner library had the following options:
- 1-5 Hidden Layers
- Activation functions either:
    - relu
    - tanh
- Up to 80 nodes in the input layer, and up to 40 nodes in the hidden layers
- 20 epochs
- Added a learning rate choice, which alows Keras Tuner to explore different learning rates. The learning rate is a critical hyperparameter that can significantly impact the training of the model.
- Include dropout layers to reduce overfitting by randomly setting a fraction of input units to 0 at each update during training.
- Added L2 regularization to the dense layers to help prevent overfitting.
  
The best model when ran with 60 trials produced:
- Accuracy: 0.7329
- Loss: 0.5801

Model Summary:

<img src="..\Images\hyperparameters_model2.png" width=200>

#### Optimized Model V3

[AlphabetSoupCharity_Optimization_V3.ipynb](../Notebooks/AlphabetSoupCharity_Optimization_V3.ipynb)
[AlphabetSoupCharity_Optimized_V3.keras](../Models_keras_files/AlphabetSoupCharity_Optimized_V3.keras)

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
- Accuracy: 0.7343 
- Loss: 0.5538

Model Summary:

<img src="..\Images\hyperparameters_model3.png" width=200>

#### Optimized Model V4

[AlphabetSoupCharity_Optimization_V4.ipynb](../Notebooks/AlphabetSoupCharity_Optimization_V4.ipynb)
[AlphabetSoupCharity_Optimized_V4.keras](../Models_keras_files/AlphabetSoupCharity_Optimized_V4.keras)

The third optimization run using the keras_tuner library had the following options:
- 1-5 Hidden Layers
- Activation functions either:
    - relu
    - tanh
- Up to 100 nodes in the input layer, and up to 50 nodes in the hidden layers
- 20 epochs
- Removed 'ASK_AMT' (which was predominatly $5000 and a then just one or two occurences of all other values) column, and added back in the 'NAME' column.
- Created more bins for rare occurrences in columns (specifically, created two Other bins Other1 and Other2 for the CLASSIFICATION column).
- Decreased the number of values in the Other bin for APPLICATION_TYPE (speficically, set it to v<156 rather than v<528).
  
The best model when ran with 60 trials produced:
- Accuracy: 0.7967
- Loss: 0.4702

Model Summary:

<img src="..\Images\hyperparameters_model4.png" width=200>

## Summary

The initial optimization model used five hidden layers with a number of neurons between 1 and 80 (first layer) and 1 and 40 (other layers) and activation function choice of either relu or tanh, and 20 epochs.

I made various attempts to better the accuracy of the initial optimised model: adding a learning rate choice, including dropout layers to reduce overfitting, adding L2 regularization to the dense layers to help prevent overfitting, adding more neurons to a hidden layer, adding more hidden layers, using different activation functions for the hidden layers, adding more epochs to the training regimen, adding early stopping to stop training when the validation loss does not improve. This is useful if the number of epochs is high, dropping more or fewer columns, creating more bins for rare occurrences in columns, increasing or decreasing the number of values for each bin.

Interestingly, most of these modifications did not give improvement over the initial optimization model. The one modification that did result in increasing the accuracy above 75% was adding the Name column back into the model. A possible explaination is that this reduces some of the "noise" in the oversampled data and allows the algorithm to further classify the data.

After adding the Name column back into the model, the final (i.e. V4) automatically optimized neural network trained model from the keras tuner method achieved 80% prediction accuracy with a 47% loss. This model used a tanh activation function with input node of 56 neurons and 5 hidden layers at a 16, 11, 16, 16, 21 neurons split and 20 training epochs, and a sigmoid output activation function. This model performed better than the non-automized model. Keeping the Name column was crucial in achieving and and going beyond the target. This shows the importance of the shape of your datasets before you preprocess it. 
