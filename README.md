# Machine-Learning-Deep-Learning-Frameworks
Jupyter notebooks implementing features of machine learning and neural networks. I learned much of this material and wrote much of this code while completing the Deep Learning specialization on Coursera.

## Logistic regression with a neural network mindset
LogisticRegressionFromScratch.ipynb:
- Step-by-step implementation of logistic regression, created in the style of a neural network.
- Contains functions that initialize variables, perform forward and backward propagation, optimize parameters using gradient descent, and make predictions on new data.
- Model is tested on sci-kit learn's breast cancer dataset.

## Neural network from scratch using TensorFlow
neural_network.py
- Defines class NeuralNetwork, which implements a neural network using TensorFlow.

NeuralNetworkDemo.ipynb
- Instantiates NeuralNetwork and applies to MNIST hand-written digit dataset.
- Defines functions to plot model performance and display images and classifications.

## Neural style transfer
neural_style_transfer.py
- Implements neural style transfer model with an object-oriented framework.
- Classes include:
  - Image: represents image and performs preprocessing to be in format needed by VGG19 model.
  - ComponentModel: represents basics of a model that serves as a component of the larger NSTModel.
    - ContentModel: implements content model, computes cost between generated image and content image.
    - StyleModel: implements style model, computes cost between generated image and style image.
  - NSTModel: model that minimizes weighted costs of the ContentModel and one or more StyleModel.

NeuralStyleTransfer.ipynb
- Implements NSTModel class, using VGG19 as a base model.
- Generates image that combines content image with style image.

<br>
More files and details to come.
