"""Neural style transfer model to convert image to specific artistic style

Model uses layers of a pre-trained base model (e.g., VGG19) to serve as
competing content and style models. A content model attempts to maximize the
similarity in activation of a late layer between the generated image and a
content image. A style model attempts to maximize the similarity in activation
of an intermediate layer of the base model between the generated image and a
style image. Each NSTModel has one ContentModel and one or more StyleModel.
The generated image minimizes a weighted sum of the costs of the ContentModel
and StyleModel.

    @author: Nicole Peltier
    @contact: nicole.eleni.peltier@gmail.com
    @date: September 15, 2020

    Functions:
        deprocess: reverse of preprocessing performed for VGG19

    Classes:
        Image: represents an image and performs preprocessing to be in format
            needed by VGG19 model
        ComponentModel: represents basics of a model that serves as a component
            of the larger NSTModel
        ContentModel(ComponentModel): implements content model, computes cost
            as the mean squared error between the activation produced by the
            generated image and the activation produced by the content image
        StyleModel(ComponentModel): implements style model, computes cost as
            the mean squared error between the gram matrices for the generated
            image and the style image
        NSTModel: model that minimizes weighted costs of the ContentModel and
            one or more StyleModel. Generates images for each iteration of
            training and produces final image that minimizes overall cost.

    Typical usage example:
        content_model = ContentModel(base_model, layer1, "content.jpg")
        style_model = StyleModel(base_model, layer2, "style.jpg")
        nst = NSTModel()
        nst.set_content_model(content_model)
        nst.add_style_model(style_model)
        nst.training_loop()
"""

# Import necessary libraries
import time
import PIL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.models import Model

def deprocess(img):
    """
    Perform reverse of VGG19's preprocessing on image

    Parameters:
        img (np.ndarray): array containing image

    Returns:
        img (np.ndarray): image array after reversal of preprocessing
    """

    # Numbers are reverse of preprocessing
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]

    # Clip values between 0 and 255, cast as int
    img = np.clip(img, 0, 255).astype('uint8')

    return img

class Image:
    """
    Image class: represent, preprocess, display, save image

    Attributes:
        _image: np.ndarray containing image (user can get attribute with img.image)
    """

    def __init__(self, filepath=None):
        """
        Initialize Image instance

        Parameters:
            filepath (str): path to image, including file extension
                            (optional, default = None)

        Returns:
            None
        """

        self._image = None
        self._unprocessed = None

        # If file path for image specified, load image
        if filepath is not None:
            self.load_image(filepath)

    def load_image(self, filepath):
        """
        Given path to image file, load image and preprocess

        Parameters:
            filepath (str): path to image, including file extension

        Returns:
            None
        """

        # Load image and perform preprocessing
        img = load_img(filepath)
        img = img_to_array(img)
        self._unprocessed = img.astype(int)

        img = preprocess_input(img)
        # Convert to 1 x n_pxl x n_pxl x n_channels
        img = np.expand_dims(img, axis=0)

        # Set image
        self._image = img

    def set_image(self, img):
        """
        Set image property to be an array passed as input

        Parameters:
            img (np.ndarray): array containing image

        Returns:
            None
        """

        self._image = img

    def display(self):
        """ Display image (no input or output) """

        # Create a copy of self._image
        img = np.copy(self._image)
        # If image is 4D, squeeze
        if len(img.shape) == 4:
            img = np.squeeze(img, axis=0)

        # Reverse preprocessing
        img = deprocess(img)

        # Use matplotlib to show image
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)

    def save(self, savedir):
        """
        Save image to file

        Parameters:
            savedir (str): path to image, including file extension

        Returns:
            None
        """

        # Create copy of image
        img = np.copy(self._image)
        # If image is 4D, squeeze
        if len(img.shape) == 4:
            img = np.squeeze(img, axis=0)

        # Reverse preprocessing
        img = deprocess(img)

        # Convert to PIL Image and save
        img = PIL.Image.fromarray(img.astype('uint8'))
        img.save(savedir)

    @property
    def image(self):
        return self._image

class ComponentModel:
    """
    ComponentModel class: define model that makes up one part of NSTModel.
    Classes ContentModel and StyleModel inherit from ComponentModel.

    Attributes:
        _base_model (TensorFlow Functional): model off of which _model is based
                (see related Jupyter notebook for implementation with VGG19)
        _layer (str): layer of base model (e.g., 'block1_conv1')
        _model (TensorFlow Functional): model to calculate activation of network
                in response to image
    """

    def __init__(self, base_model=None, layer=None, imagepath=None):
        """
        Initialize ComponentModel instance

        Parameters:
            base_model (TensorFlow model): optional, default = None
            layer (str): layer of base model (e.g., 'block1_conv1'), optional
                        (default = None)
            imagepath (str): path to image, including file extension, optional
                        (default = None)

        Returns:
            None
        """

        self._base_model = None
        self._layer = None
        self._model = None

        if base_model is not None:
            self._base_model = base_model
        if layer is not None:
            self.set_layer(layer)
        if imagepath is not None:
            self.set_image(imagepath)

    def set_layer(self, layer):
        """
        Set layer of model according to base model

        Parameters
            layer (str): layer of base model (e.g., 'block1_conv1')

        Returns
            None
        """

        self._layer = layer
        self.set_model()

    def set_model(self):
        """
        Set model that calculates activation of network in response to image

        Returns:
            None
        """
        self._model = Model(inputs=self._base_model.input,
                            outputs=self._base_model.get_layer(self._layer).output)

    def set_image(self, imagepath):
        """
        Set image associated with content model

        Parameters:
            imagepath (str): path to image, including file extension

        Returns:
            None
        """

        self._image = Image(filepath=imagepath)

    @property
    def layer(self):
        return self._layer

    @property
    def base_model(self):
        return self._base_model
    
    @property
    def image(self):
        return self._image

class ContentModel(ComponentModel):
    """
    ContentModel class: model that attempts to minimize difference in
    activation between example content image and model-generated image
    """

    def __init__(self, base_model=None, layer=None, imagepath=None):
        """
        Initialize ContentModel instance

        Parameters:
            base_model (TensorFlow Functional): model off of which _model is
                        based, optional (default = None)
            layer (str): layer of base model (e.g., 'block1_conv1'), optional
                        (default = None)
            imagepath (str): path to image, including file extension, optional
                        (default = None)

        Returns:
            None
        """

        super().__init__(base_model, layer, imagepath)

    def cost(self, generated):
        """
        Compute cost of content model as mean squared error between activation
        of content image and activation of generated image

        Parameters:
            generated (tf ResourceVariable containing
                       np.ndarray): model-generated image

        Returns:
            cost (float): cost of content model
        """

        # Calculate activation
        activation_content = self._model(self._image.image)
        activation_generated = self._model(generated)

        # Compute cost as mean squared error
        cost = tf.reduce_mean(tf.square(activation_content - activation_generated))
        return cost

class StyleModel(ComponentModel):
    """
    StyleModel class: model that attempts to minimize difference in
    activation between example style image and model-generated image. Note:
    Full NSTModel may have multiple StyleModel.
    """

    def __init__(self, base_model=None, layer=None, imagepath=None):
        """
        Initialize StyleModel instance

        Parameters:
            base_model (TensorFlow Functional): model off of which _model is
                        based, optional (default = None)
            layer (str): layer of base model (e.g., 'block1_conv1'), optional
                        (default = None)
            imagepath (str): path to image, including file extension, optional
                        (default = None)

        Returns:
            None
        """

        super().__init__(base_model, layer, imagepath)

    def gram_matrix(self, A):
        """
        Compute gram matrix of model activation

        Parameters
            A (np.ndarray): activation matrix

        Returns
            gram matrix of A (np.ndarray)
        """

        n_C = int(A.shape[-1])
        a = tf.reshape(A, [-1, n_C])
        n = tf.shape(a)[0]
        G = tf.matmul(a, a, transpose_a=True)
        return G / tf.cast(n, tf.float32)

    def cost(self, generated):
        """
        Compute cost of style model as mean squared error between gram matrices
        of activation of content image and activation of generated image

        Parameters:
            generated (tf ResourceVariable containing
                       np.ndarray): model-generated image

        Returns:
            cost (float): cost of content model
        """

        # Compute activation of model by style image and generated image
        activation_style = self._model(self._image.image)
        activation_generated = self._model(generated)

        # Compute gram matrix of style image activation and generated image activation
        gram_style = self.gram_matrix(activation_style)
        gram_generated = self.gram_matrix(activation_generated)

        # Compute mean squared error
        cost = tf.reduce_mean(tf.square(gram_style - gram_generated))

        return cost

class NSTModel:
    """
    NSTModel class: Generates images by minimizing weighted costs of the
            ContentModel and one or more StyleModel.
    
    Attributes:
        _content_model (ContentModel): model that minimizes loss betweeen 
                        generated image and content image
        _style_models (list of StyleModel): models that minimze loss between
                        generated image and style image
        _style_weights (list of floats): weight of each style model
        _alpha (float): relative weight of content model
        _beta (float): relative weight of style models
        _iterations (int): number of iterations of model training
        _generated_images (list of Image): images generated for each iteration
                        of model training
        _best_image (Image): image that minimizes overall loss with respect to
                        both style and content models
    """

    def __init__(self, alpha=10., beta=20.):
        """
        Initialize NSTModel instance

        Parameters
            alpha (float): weight of content model, optional (default = 10)
            beta (float): weight of style models, optional (default = 20)

        Returns
            None
        """

        self._content_model = None
        self._style_models = []
        self._style_weights = None
        self._alpha = alpha
        self._beta = beta
        self._iterations = 5
        self._generated_images = []
        self._best_image = None

    def set_alpha(self, alpha):
        """ Set weight of content model to be alpha """
        self._alpha = alpha

    def set_beta(self, beta):
        """ Set weight of style model to be beta """
        self._beta = beta

    def set_iterations(self, it):
        """ Set number of iterations to train model """
        self._iterations = it

    def set_content_model(self, content_model):
        """ Set content model """
        assert isinstance(content_model, ContentModel)
        self._content_model = content_model

    def reset_style_models(self):
        """ Remove all style models """
        self._style_models = []

    def add_style_model(self, style_model):
        """ Add style model """
        assert isinstance(style_model, StyleModel)
        self._style_models.append(style_model)
        self.set_style_model_weights()

    def set_style_model_weights(self, w=None):
        """
        Set relative weights of each style model

        Parameters:
            w (list of floats): weight of each style model, optional
                    (default = None, which assigns equal weight to all models)

        Returns:
            None
        """

        n_style_models = len(self._style_models)
        if w == None: # if no weights provided, assign equal weights to all
            self._style_weights = [1.0 / n_style_models] * n_style_models
        else: # if weights provided, set respective weight for each model
            assert len(w) == len(self._style_models)
            self._style_weights = w

    def content_cost(self, generated):
        """
        Compute cost of content model

        Parameters
            generated (tf ResourceVariable containing
                       np.ndarray): model-generated image

        Returns
            cost (float): cost of content model
        """
        return self._content_model.cost(generated)

    def style_cost(self, generated):
        """
        Compute cost of all style models as weighted sum of each model's cost

        Parameters
            generated (tf ResourceVariable containing
                       np.ndarray): model-generated image

        Returns
            cost (float): weighted sum cost across all style models
        """
        cost = 0
        for ind, style_model in enumerate(self._style_models):
            w = self._style_weights[ind]
            cost += w * style_model.cost(generated)
        return cost

    def compute_cost(self, generated):
        """
        Compute cost of model as weighted average of content cost and style cost

        Parameters
            generated: model-generated image

        Returns
            cost: alpha * content_cost + beta * style_cost
        """
        style = self.style_cost(generated)
        content = self.content_cost(generated)
        cost = self._alpha * content + self._beta * style
        return cost

    def training_loop(self):
        """ Train model and generate stylized image """

        generated = tf.Variable(self._content_model.image.image, dtype=tf.float32)
        self._generated_images = []

        opt = tf.optimizers.Adam(learning_rate=7.)

        best_cost = 1e12 + 0.1
        best_image = None

        start_time = time.time()

        for i in range(self._iterations):
            with tf.GradientTape() as tape:
                J_total = self.compute_cost(generated)

            grads = tape.gradient(J_total, generated)

            opt.apply_gradients([(grads, generated)])

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated.numpy() # convert from tensor to numpy array

            print(f'Cost at {i}: {J_total}. Time elapsed: {time.time() - start_time}')

            img = Image()
            img._image = generated.numpy()
            self._generated_images.append(img)

        best_img = Image()
        best_img.set_image(best_image)
        self._best_image = best_img

    def show_results(self):
        """ Display content image, style image, and generated image in one plot """

        # Create figure
        plt.figure(figsize=(15, 5))

        # Display content image
        plt.subplot(1, 3, 1)
        self.content_model.image.display()
        plt.title('Content image')

        # Display style image
        plt.subplot(1, 3, 2)
        self.style_models[0].image.display()
        plt.title('Style image')

        # Display generated image
        plt.subplot(1, 3, 3)
        self.best_image.display()
        plt.title('Generated image')

        plt.show()

    def show_generated_images(self):
        """ Display sequence of generated images over model training iterations """

        # Compute number of images to determine size of figure and number of rows of subplot
        n_imgs = len(self._generated_images)
        n_rows = np.ceil(n_imgs / 5)
        plt.figure(figsize = (12, n_rows * 3))

        # Loop through images and display
        for i in range(n_imgs):
            plt.subplot(n_rows, 5, i + 1)
            self._generated_images[i].display()
            plt.title(f'Iteration {i}')

        plt.tight_layout()
        plt.show()

    @property
    def content_model(self):
        return self._content_model

    @property
    def style_models(self):
        return self._style_models

    @property
    def style_weights(self):
        return self._style_weights

    @property
    def generated_images(self):
        return self._generated_images

    @property
    def best_image(self):
        return self._best_image
