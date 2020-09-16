# Neural Style Transfer

[Neural style transfer](https://en.wikipedia.org/wiki/Neural_Style_Transfer) uses neural networks to manipulate an image in order to emulate the style of another image.
A neural style transfer model generates an image that balances the fitting of a content model and one or more style models. The content model is trained to minimize the
difference between the generated image and a content image, while the style models are trained to minimize the difference between the generated image and a style image.
This project created content and style models using the pretrained [VGG19 model](https://keras.io/api/applications/vgg/). The activation of each model was computed as the
output of one layer of VGG19. Typically, a content model's activation will be the output of a later layer of the larger neural network, while a style model's activation
may be from an early or intermediate layer of the network.

## neural_style_transfer.py

**neural_style_transfer.py** implements a neural style transfer model with an object-oriented framework.

Classes include:
- Image: represents image and performs preprocessing to be in format needed by VGG19 model.
- ComponentModel: represents basics of a model that serves as a component of the larger NSTModel.
  - ContentModel: implements content model, computes cost between generated image and content image.
  - StyleModel: implements style model, computes cost between generated image and style image.
- NSTModel: model that minimizes weighted costs of the ContentModel and one or more StyleModel.

## NeuralStyleTransfer.ipynb

**NeuralStyleTransfer.ipynb** implements the NSTModel class and demonstrates its use. The model starts with a content image:

![Content image](https://github.com/NicoleEleniPeltier/Machine-Learning-Deep-Learning-Frameworks/blob/master/Neural-Style-Transfer/content.jpg)

and a style image:

![Style image](https://github.com/NicoleEleniPeltier/Machine-Learning-Deep-Learning-Frameworks/blob/master/Neural-Style-Transfer/style.jpg)

and it generates a new image that has the same content of the first image and replicates the style of the second image:

![Generated image](https://github.com/NicoleEleniPeltier/Machine-Learning-Deep-Learning-Frameworks/blob/master/Neural-Style-Transfer/generated.jpg)
