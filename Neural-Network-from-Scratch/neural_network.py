import tensorflow as tf
import numpy as np

class NeuralNetwork:

    def __init__(self, layers):
        """ Initialize NeuralNetwork class
                Input:
                    layers: list of number of units per layer, including input features
                            and output units
                Attributes:
                    layers: list of number of units per layer, including input features
                            and output units
                    L: number of layers
                    num_features: number of input features
                    num_classes: number of classes
                    W: dictionary containing network weights
                    b: dictionary containing network bias terms
                    dW: dictionary containing gradient of loss with respect to W
                    db: dictionary containing gradient of loss with respect to b
        """
        # Extract details about network from layers
        self.layers = layers
        self.L = len(layers)
        self.num_features = layers[0]
        self.num_classes = layers[-1]

        # Declare empty dictionaries for W and b
        self.W = {}
        self.b = {}

        # Declare empty dictionaries for dW and db
        self.dW = {}
        self.db = {}

        # Initialize W and b randomly
        self.setup()

    def setup(self):
        """ Initialize W and b for each layer of network
                W[a] dimensions: (n_units[a], n_units[a-1])
                b[a] dimensions: (n_units[a], 1)
        """
        # Loop through layers, randomly initialize parameters
        for i in range(1, self.L):
            self.W[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

    def forward_pass(self, X):
        """ Perform forward propagation
                Input:
                    X: input data (n_samples * n_features)
                Output:
                    A: activation of last layer of network, before softmax
        """
        # Convert X to tensor before forward propagation
        A = tf.convert_to_tensor(X, dtype=tf.float32)

        # Loop through layers and compute output
        for i in range(1, self.L):
            # Compute output of layer of model given inputs and W, b
            Z = tf.matmul(A, tf.transpose(self.W[i])) + tf.transpose(self.b[i])

            # Apply linear rectifier to all layers except last one
            if i != self.L-1:
                A = tf.nn.relu(Z)
            else:
                A = Z

        return A

    def compute_loss(self, A, Y):
        """ Compute average loss of model
                Input:
                    A: predicted class labels
                    Y: true class labels
                Output: loss
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(Y, A)
        return tf.reduce_mean(loss)

    def update_parameters(self, lr):
        """ Update parameters W,b given gradients dW, db
                Input:
                    lr: learning rate
        """
        for i in range(1, self.L):
            self.W[i].assign_sub(lr * self.dW[i]) # W -= ...
            self.b[i].assign_sub(lr * self.db[i])

    def predict(self, X):
        """ Predict class for input data
                Input:
                    X: input data (n_samples * n_features)
                Output: most likely class for samples in X
        """
        A = self.forward_pass(X)
        return tf.argmax(tf.nn.softmax(A), axis=1)

    def info(self):
        """ Print information about model
        """
        num_params = 0
        for i in range(1, self.L):
            num_params += self.W[i].shape[0] * self.W[i].shape[1]
            num_params += self.b[i].shape[0]
        print('Input Features:', self.num_features)
        print('Number of Classes:', self.num_classes)
        print('Hidden Layers:')
        print('--------------')
        for i in range(1, self.L-1):
            print('Layer {}, Units {}'.format(i, self.layers[i]))
        print('--------------')
        print('Number of parameters:', num_params)

    def train_on_batch(self, X, Y, lr):
        """ Train model on (mini-)batch of data, computing gradient of loss
            with respect to W and b (dW, db) and updating W and b accordingly
                Input:
                    X: input data (n_samples * n_features)
                    Y: class labels for data in X
                    lr: learning rate
                Output: loss on training data
        """
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            A = self.forward_pass(X)
            loss = self.compute_loss(A, Y)

        # Compute gradient of loss with respect to W and b
        for i in range(1, self.L):
            self.dW[i] = tape.gradient(loss, self.W[i])
            self.db[i] = tape.gradient(loss, self.b[i])

        del tape

        # Update parameters with new values for dW and db
        self.update_parameters(lr)

        return loss.numpy()

    def train(self, x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr):
        """ Predict class for input data
                Input:
                    x_train:
                    y_train:
                    x_test:
                    y_test:
                    epochs:
                    steps_per_epoch:
                    batch_size:
                    lr: learning rate
                Output: most likely class for samples in X
        """
        history = {
            'val_loss': [],
            'train_loss': [],
            'val_acc': []
        }

        for e in range(epochs):
            epoch_train_loss = 0
            print('Epoch {}'.format(e), end='.')
            for i in range(steps_per_epoch):
                x_batch = x_train[i*batch_size: (i+1)*batch_size]
                y_batch = y_train[i*batch_size: (i+1)*batch_size]

                batch_loss = self.train_on_batch(x_batch, y_batch, lr)
                epoch_train_loss += batch_loss

                if i%int(steps_per_epoch/10) == 0:
                    print(end='.')

            history['train_loss'].append(epoch_train_loss/steps_per_epoch)
            val_A = self.forward_pass(x_test)
            val_loss = self.compute_loss(val_A, y_test).numpy()
            history['val_loss'].append(val_loss)
            val_preds = self.predict(x_test)
            val_acc = np.mean(np.argmax(y_test, axis=1) == val_preds.numpy())
            history['val_acc'].append(val_acc)
            print('Val acc:', val_acc)

        return history
