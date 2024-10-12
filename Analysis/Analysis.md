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

I build the first model with the following parameters with low computation time in mind:

- 2 hidden layers with 80, 30 neurons split (the input (node) feature was 43, 80 was chosen as the first layer as it is almost double the input_feature). With an hidden layer activation function of relu as this our go to for first model.

- Output node is 1 as it was binary classifier model with only one output: was the funding application succesfull yes or no. And an output layer activation of sigmoid as the model output is binary classification between 0 and 1.
- 
I then increased the hidden layers to 3 and set the third hidden layer at 30 as the model prediction accuracy was below 75%: 

For the second model I decided to use tanh activation and 3 hidden layers with 90, 30, 20 neurons split and a sigmoid activation for output as the output doesn't change.

I experimented with increasing nodes and neurons, with changing other parameters to get a better accuracy but despite doing this both models came below the 75% threshold.

### 3: Optimize the Model

I decided to use an automated model optimizer to get the most accurate model possible by creating method that creates a keras Sequential model using the keras-tuner library with hyperparametes options.


## Further and Final Optimization

I kept the Name column for my final Optimized Model as I still hadn't reached the goal of 75% accuracy. Kepping the keras-tuner the same apart from lowering the epochs from 100 to 50 for time optimization.

## Summary:

The final automatically optimized neural network trained model from the keras tuner method achieved 80% prediction accuracy with a 0.45 loss, using a sigmoid activation function with input node of 76; 5 hidden layers at a 16, 21, 26, 11, 21, neurons split and 50 training epochs. Performing better than the non automized model. Keeping the Name column was crucial in achieving and and going beyond the target. This shows the importance of the shape of your datasets before you preprocess it.

Overall we were able to consistently train and test a neural network which can accurrately predict the success outcome of a funding venture 72.5% of the time. It would fall upon the risk tolerance of the Alphabet Soup company to decide if this 72.5% accuracy is close enough to the target 75% for them to move forward with implementing the model in their decision making process.

In reseraching I learned others were able to reach an accuracy of over 80% by including the 'NAME' feature back into the algorithm and normalizing it. The explanation being that it reduces some of the "noise" in the oversampled data and allows the algorithm to further classify the data.

In the future I would change the model to try a different input activation and increase the number of nodes and hidden layers. Often the shape of the input data can be an important factor in the accuracy and success of a model. While increasing the number of hidden layers and neurons in each layer can improve the overall accuracy as well.


Summary of Results

Overall the deep learning model did give near 73% accuracy on a dataset with many dimensions and with some complexity.
Whilst this didn't quite meet the desired performance goal of 75%, the optimization resulted in a decrease for the number of neuron's required with only one additional layer required, compared to my initial model

Alternative Model

An alternative model to the one used within my analysis is the Random Forest algorithm
This model is widely used within the finance industry
The model can be easily scaled to large datasets, is typically good at not overfitting to data, and is resillient to noisy data
The random forest model works by creating random predictions from decision trees and creates an average of these results to build a model
