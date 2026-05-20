'''network.py
Represents  a neural network (collection of layers)
GLORY MAHENGE AND OPEOLUWA SHODIPE
Neural Networks
Convolutional Neural Networks
'''
import time
import numpy as np

import layer
import filter_ops

import accelerated_layer as acc


class Network:
    '''Represents a neural network with some number of layers of various types.
    To create a specific network, create a subclass (e.g. ConvNet4) then
    add layers to it. For this project, the focus will be on building the
    ConvNet4 network.
    '''
    def __init__(self, reg=0, verbose=True):
        '''Method is pre-filled for you (shouldn't require modification).'''
        # Python list of Layer object references that make up out network
        self.layers = []
        # Regularization strength
        self.reg = reg
        # Whether we want network-related debug/info print outs
        self.verbose = verbose

        # Python list of ints. These are the indices of layers in `self.layers`
        # that have network weights.
        self.wt_layer_inds = []

        # Are we currently training?
        self.training = False

        # As in former projects, Python list of loss, training/validation
        # accuracy during training recorded at some frequency (e.g. every epoch)
        self.loss_history = []
        self.train_acc_history = []
        self.validation_acc_history = []

    def accuracy(self, inputs, y, samp_sz=500, mini_batch_sz=64):
        '''Computes accuracy using current net on the inputs `inputs` with classes `y`.

        This method is pre-filled for you (shouldn't require modification).

        Parameters:
        ----------
        inputs: ndarray. shape=shape=(num samples, n_chans, img_y, img_x)
            We are testing the classification accuracy on these data.
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(N,) for mini-batch size N.
        samp_sz: int. If the number of samples is bigger than this number,
            we take a random sample from `inputs` of this size. We do this to
            keep performance of this method reasonable.
        mini_batch_sz: Because it might be tricky to hold all the training
            instances in memory at once, process and evaluate the accuracy of
            samples from `input` in mini-batches. We merge the accuracy scores
            across batches so the result is no different than processing all at
            once.
        '''
        n_samps = len(inputs)

        # Do we subsample the input?
        if n_samps > samp_sz:
            rng = np.random.default_rng(0)
            subsamp_inds = rng.choice(n_samps, samp_sz)
            n_samps = samp_sz
            inputs = inputs[subsamp_inds]
            y = y[subsamp_inds]

        # How many mini-batches do we split the data into to test accuracy?
        n_batches = int(np.ceil(n_samps / mini_batch_sz))
        # Placeholder for our predicted class ints
        y_pred = np.zeros(len(inputs), dtype=np.int32)

        # Compute the accuracy through the `predict` method in batches.
        # Strategy is to use a 1D cursor `b` to extract a range of inputs to process (a mini-batch)
        for b in range(n_batches):
            low = b*mini_batch_sz
            high = b*mini_batch_sz+mini_batch_sz
            # Tolerate out-of-bounds as we reach the end of the num samples
            if high > n_samps:
                high = n_samps

            # Get the network predicted classes and put them in the right place
            y_pred[low:high] = self.predict(inputs[low:high])

        # Accuracy is the average proportion that the prediction matchs the true int class
        acc = np.mean(y_pred == y)

        return acc

    def compile(self, optimizer_name, **kwargs):
        '''Tells each network layer how weights should be updated during backprop
        during training (e.g. stochastic gradient descent, adam, etc.)

        This method is pre-filled for you (shouldn't require modification).

         This NEEDS to be called AFTER creating your ConvNet4 object,
        but BEFORE you call `fit()` to train the net (otherwise, how does your
        net know how to update the weights?).

        Parameters:
        ----------
        optimizer_name: string. Name of optimizer class to use to update wts.
            See optimizer::create_optimizer for specific ones supported.
        **kwargs: Any number of optional parameters that get passed to the
            optimizer of your choice. e.g. learning rate.
        '''
        # Only set an optimizer for each layer with weights
        for l in [self.layers[i] for i in self.wt_layer_inds]:
            l.compile(optimizer_name, **kwargs)

    def forward(self, inputs, y):
        '''Do forward pass through whole network

        Parameters:
        ----------
        inputs: ndarray.
            Inputs coming into the input layer of the net. shape=(B, n_chans, img_y, img_x)
        y: ndarray.
            int-coded class assignments of training mini-batch. 0,...,numClasses-1 shape=(B,) for mini-batch size B.

        Returns:
        -------
        loss: float.
            REGULARIZED loss.

        Outline:
        1. Call the forward method of each layer in the network.
            Make the output of the previous layer the input to the next.
        2. Compute and get the loss from the LAST network layer.
        2. Compute and get the weight regularization via `self.wt_reg_reduce()` (implement this next)
        4. Return the sum of the loss and the regularization term.
        '''
        x = inputs

        for layer in self.layers:
            x = layer.forward(x)

        loss = self.layers[-1].loss(y)

        reg_loss = self.wt_reg_reduce()

        return loss + reg_loss
     


    def wt_reg_reduce(self):
        '''Computes the loss weight regularization for all network layers that have weights

        Returns:
        -------
        wt_reg: float. Regularization for weights from all layers across the network.

        Layer indicies with weights are maintained in `self.wt_layer_inds`.
        The network regularization `wt_reg` is simply the sum of all the regularization terms
        for each individual layer.
        '''
        wt_reg = 0.0
    
        for i in self.wt_layer_inds:
            layer = self.layers[i]
            wt_reg += 0.5 * layer.reg * np.sum(layer.wts ** 2)
        
        return wt_reg
        

    def backward(self, y):
        '''Initiates the backward pass through all the layers of the network.

        Parameters:
        ----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -------
        None

        Outline:
        1. Initialize d_upstream, d_wts, d_b to None.
        2. Loop through the network layers in REVERSE ORDER, calling the `Layer` backward method.
            Remember that the output of layer.backward() becomes the d_upstream to the next layer down.
            We don't care about d_wts, d_b in this method (computed/stored in Layer).
        '''
        d_upstream = None
        d_wts = None
        d_b = None

        for layer in reversed(self.layers):
            # d_net_in = layer.backward_netAct_to_netIn(d_upstream, y)
            # dprev_net_act, d_net_wts, d_b = layer.backward_netIn_to_prevLayer_netAct(d_net_in)
            # d_upstream = dprev_net_act

            dprev_net_act, d_wts, d_b = layer.backward(d_upstream, y)
            d_upstream = dprev_net_act


    def is_training(self):
        '''Is the CNN currently training?

        (Week 4)

        Returns:
        --------
        bool. Whether net is training.
        '''
        return self.training

    def set_training_mode(self, training=False):
        '''Relays whether we are in training mode to the network layers.
        This information only changes the behavior Dropout layers.

        (Week 4)

        Parameters:
        -----------
        in_train_mode: bool.
            Are we currently training the net?

        Outline:
        1. Update the network's mode.
        2. Update each of the layer's mode.
        '''
        self.training = training
        for layer in self.layers:
            layer.in_train_mode = training

    def predict(self, inputs):
        '''Do forward pass thru the net and classfies novel inputs presented to the network.

         It may NOT make sense to call's network's forward method...

        Parameters:
        ----------
        inputs: ndarray. shape=shape=(num test samples, n_chans, img_y, img_x)
            This is test data.

        Returns:
        -------
        pred_classes: ndarray. shape=shape=(num test samples)
            Predicted classes (int coded) derived from the network.
        '''
        x = inputs
    
        og_mode = self.training 
        self.set_training_mode(False)

        for layer in self.layers:
            x = layer.forward(x)
        
        self.set_training_mode(og_mode)
        pred_classes = np.argmax(x, axis=1)
        
        return pred_classes  

    def fit(self, x_train, y_train, x_validate, y_validate, mini_batch_sz=256, n_epochs=1, print_every=10):
        '''Trains the neural network on data

        Parameters:
        ----------
        x_train: ndarray. shape=(num training samples, n_chans, img_y, img_x).
            Training data.
        y_train: ndarray. shape=(num training samples,).
            Training data classes, int coded.
        x_validate: ndarray. shape=(num validation samples, n_chans, img_y, img_x).
            Every so often during training (see acc_freq param), we compute
            the accuracy of the network in classifying the validation set
            (out-of-training-set generalization). This is the data we use.
        y_validate: ndarray. shape=(num validation samples,).
            Validation data classes, int coded.
        mini_batch_sz: int.
            Mini-batch training size.
        n_epochs: int.
            Number of training epochs.
        print_every: int.
            Controls the frequency (in iterations/mini-batches) with which to wait before printing out the
            loss, accuracy, and iteration/mini-batch number.
            
            - Previously, you used number of epochs rather than iterations to measure the frequency
            of print-outs. Use the simpler-to-implement units of iterations here because CNNs are
            more computationally intensive and you may want some print-outs during an epoch.
            - It is a computationally intensive process for the CNN to compute training and validation accuracies.
            So only compute training and validation set accuracies on iterations when you print.
            (i.e DON'T compute them every iteration).

        Returns:
        --------
        self.loss_history: list.
            Record of training loss computed over each mini-batch of training for every epoch.
        self.train_acc_history. list.
            Record of accuracy on training set computed on entire training set computed every `print_every` training iters.
        self.validation_acc_history
            Record of accuracy on val set computed on val training set computed every `print_every` training iters.

        Outline: Complete this method's implementation.
        - It should follow the familar format for a training loop: Randomly sampling to get a mini-batch,
        doing the forward and backward pass, computing the loss, updating the wts/biases.
            -  There is a provided weight update method in Layer called `update_weights`.
        - Add support for `print_every`.
        - Use the Python time module to print out the runtime (in minutes) for iteration 0 only.
        Also printout the projected time for completing ALL training iterations. (For simplicity, you don't need to
        consider the time taken for computing train and validation accuracy).
        '''
        n_samples = len(x_train)
        iteration = 0

        self.set_training_mode(True)
        
        for epoch in range(n_epochs):
            # Shuffle training data each epoch
            rng = np.random.default_rng()

            # Mini-batch loop
            n_batches = int(np.ceil(n_samples / mini_batch_sz))
            total_iters = n_epochs * n_batches
            
            for b in range(n_batches):
                #  mini-batch
                low = b * mini_batch_sz
                high = min(low + mini_batch_sz, n_samples)
                x_batch = x_train[low:high]
                y_batch = y_train[low:high]
                
                # Time iter 0
                if iteration == 0:
                    start = time.time()
                
                loss = self.forward(x_batch, y_batch)
                self.loss_history.append(loss)
                
                self.backward(y_batch)
                
                for i in self.wt_layer_inds:
                    self.layers[i].update_weights()
                
                if iteration % print_every == 0 or iteration == total_iters - 1:
                    train_acc = self.accuracy(x_train, y_train)
                    val_acc = self.accuracy(x_validate, y_validate)
                    
                    self.train_acc_history.append(train_acc)
                    self.validation_acc_history.append(val_acc)
                    
                    print(f"Finished Iter {iteration} / {total_iters}. Loss: {loss}")
                    print(f"Train acc: {train_acc} , Val acc: {val_acc}")
                    
                    if iteration == 0:
                        elapsed = time.time() - start
                        projected = (elapsed * total_iters) / 60
                        print(f"{elapsed} sec to finish 1 iter. Estimated total time: {projected} mins.")

                iteration += 1
        
        self.set_training_mode(False)

        return self.loss_history, self.train_acc_history, self.validation_acc_history


class ConvNet4(Network):
    '''A minimal convolutional neural network.
    Makes a ConvNet4 network with the following layers: Conv2D -> MaxPool2D -> Flatten -> Dense -> Dense

    0. Convolution (net-in), Relu (net-act).
    1. Max pool 2D (net-in), linear (net-act).
    2. Flatten (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-2, reg=0, r_seed=None,
                 verbose=True):
        '''ConvNet4 constructor. The job of this method is to build the network as a collection of connected layers
        (in order) in `self.layers`.

        Parameters:
        -----------
        input_shape: tuple. By default: (n_chans, img_y, img_x)
            Shape of a SINGLE input sample (no mini-batch).
        n_kers: tuple.
            Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple rather than just an
            int. The reasoning is that if you wanted to create another Conv2D layer, say with 16 units, n_kers would
            then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple.
            x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST Conv2D layer.
            If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Same format as above.
            Number of hidden units in each dense layer.
             Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple.  Same format as above.
            Pooling extent in the i-th MaxPool2D layer.
        pooling_strides: tuple.  Same format as above.
            Pooling stride in the i-th MaxPool2D layer.
        n_classes: int.
            Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float.
            Global weight scaling to use for all layers with weights
        reg: float.
            Regularization strength
        r_seed: int or None.
            Random seed for setting weights and bias parameters.
        verbose: bool. Do we want to term network-related debug print outs on?
             This is different than per-layer verbose settings, which are turned manually on below.

        
        - Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        - Number your layers starting at 0.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

       
        self.layers.append(layer.Conv2D(0, 'Conv2D', n_kers=n_kers[0], ker_sz=ker_sz[0], n_chans=n_chans, wt_scale=wt_scale, activation='relu', reg=reg, r_seed=r_seed, verbose=False))


        self.layers.append(layer.MaxPool2D(1, 'MaxPool2D', pool_size=pooling_sizes[0], strides=pooling_strides[0], activation='linear', verbose=False))

        self.layers.append(layer.Flatten(2, 'Flatten', verbose=False))

        #flattened size for Dense layer input
        pool_out_y = (h - pooling_sizes[0])//pooling_strides[0] + 1
        pool_out_x = (w - pooling_sizes[0]) // pooling_strides[0] + 1
        flat_units = n_kers[0] * pool_out_y * pool_out_x

        self.layers.append(layer.Dense(3, 'Dense_hidden', units=dense_interior_units[0], n_units_prev_layer=flat_units, wt_scale=wt_scale, activation='relu', reg=reg, r_seed=r_seed, verbose=False))

        self.layers.append(layer.Dense(4, 'Dense_output', units=n_classes, n_units_prev_layer=dense_interior_units[0], wt_scale=wt_scale, activation='softmax', reg=reg, r_seed=r_seed, verbose=False))

        self.wt_layer_inds = [0, 3, 4]

class ConvNet4Accel(Network):
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-2, reg=0, r_seed=None,
                 verbose=True):
        '''ConvNet4 constructor. The job of this method is to build the network as a collection of connected layers
        (in order) in `self.layers`.

        Parameters:
        [p[]
        -----------
        input_shape: tuple. By default: (n_chans, img_y, img_x)
            Shape of a SINGLE input sample (no mini-batch).
        n_kers: tuple.
            Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple rather than just an
            int. The reasoning is that if you wanted to create another Conv2D layer, say with 16 units, n_kers would
            then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple.
            x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST Conv2D layer.
            If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Same format as above.
            Number of hidden units in each dense layer.
             Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple.  Same format as above.
            Pooling extent in the i-th MaxPool2D layer.
        pooling_strides: tuple.  Same format as above.
            Pooling stride in the i-th MaxPool2D layer.
        n_classes: int.
            Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float.
            Global weight scaling to use for all layers with weights
        reg: float.
            Regularization strength
        r_seed: int or None.
            Random seed for setting weights and bias parameters.
        verbose: bool. Do we want to term network-related debug print outs on?
             This is different than per-layer verbose settings, which are turned manually on below.

        
        - Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        - Number your layers starting at 0.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

       
        self.layers.append(acc.Conv2DAccel(0, 'Conv2D', n_kers=n_kers[0], ker_sz=ker_sz[0], n_chans=n_chans, wt_scale=wt_scale, activation='relu', reg=reg, r_seed=r_seed, verbose=False))


        self.layers.append(acc.MaxPool2DAccel(1, 'MaxPool2D', pool_size=pooling_sizes[0], strides=pooling_strides[0], activation='linear', verbose=False))

        self.layers.append(layer.Flatten(2, 'Flatten', verbose=False))

        #flattened size for Dense layer input
        pool_out_y = (h - pooling_sizes[0])//pooling_strides[0] + 1
        pool_out_x = (w - pooling_sizes[0]) // pooling_strides[0] + 1
        flat_units = n_kers[0] * pool_out_y * pool_out_x

        self.layers.append(layer.Dense(3, 'Dense_hidden', units=dense_interior_units[0], n_units_prev_layer=flat_units, wt_scale=wt_scale, activation='relu', reg=reg, r_seed=r_seed, verbose=False))

        self.layers.append(layer.Dense(4, 'Dense_output', units=n_classes, n_units_prev_layer=dense_interior_units[0], wt_scale=wt_scale, activation='softmax', reg=reg, r_seed=r_seed, verbose=False))

        self.wt_layer_inds = [0, 3, 4]

class ConvNet4AccelV2(Network):
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-2, reg=0, r_seed=None,
                 verbose=True, dropout_rate = 0.0):
        '''ConvNet4 constructor. The job of this method is to build the network as a collection of connected layers
        (in order) in `self.layers`.

        Parameters:
        [p[]
        -----------
        input_shape: tuple. By default: (n_chans, img_y, img_x)
            Shape of a SINGLE input sample (no mini-batch).
        n_kers: tuple.
            Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple rather than just an
            int. The reasoning is that if you wanted to create another Conv2D layer, say with 16 units, n_kers would
            then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple.
            x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST Conv2D layer.
            If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Same format as above.
            Number of hidden units in each dense layer.
             Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple.  Same format as above.
            Pooling extent in the i-th MaxPool2D layer.
        pooling_strides: tuple.  Same format as above.
            Pooling stride in the i-th MaxPool2D layer.
        n_classes: int.
            Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float.
            Global weight scaling to use for all layers with weights
        reg: float.
            Regularization strength
        r_seed: int or None.
            Random seed for setting weights and bias parameters.
        verbose: bool. Do we want to term network-related debug print outs on?
             This is different than per-layer verbose settings, which are turned manually on below.

        
        - Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        - Number your layers starting at 0.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        self.layers.append(acc.Conv2DAccel(0, 'Conv2D', n_kers=n_kers[0], ker_sz=ker_sz[0], n_chans=n_chans, wt_scale=wt_scale, activation='relu', reg=reg, r_seed=r_seed, verbose=False))

        self.layers.append(acc.MaxPool2DAccel(1, 'MaxPool2D', pool_size=pooling_sizes[0], strides=pooling_strides[0], activation='linear', verbose=False))

        self.layers.append(layer.Flatten(2, 'Flatten', verbose=False))

        pool_out_y = (h - pooling_sizes[0]) // pooling_strides[0] + 1
        pool_out_x = (w - pooling_sizes[0]) // pooling_strides[0] + 1
        flat_units = n_kers[0] * pool_out_y * pool_out_x

        self.layers.append(layer.Dense(3, 'Dense_hidden', units=dense_interior_units[0],
                                       n_units_prev_layer=flat_units, wt_scale=wt_scale, activation='relu',
                                       reg=reg, r_seed=r_seed, verbose=False))

        self.layers.append(layer.Dropout(4, 'Dropout', rate=dropout_rate, r_seed=r_seed, verbose=False))

        self.layers.append(layer.Dense(5, 'Dense_output', units=n_classes,
                                       n_units_prev_layer=dense_interior_units[0], wt_scale=wt_scale,
                                       activation='softmax', reg=reg, r_seed=r_seed, verbose=False))

        self.wt_layer_inds = [0, 3, 5]
