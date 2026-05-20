'''filter_ops.py
Implements the convolution and max pooling operations.
Applied to images and other data represented as an ndarray.
GLORY MAHENGE & OPEOLUWA SHODIPE
Neural Networks
Convolutional neural networks
'''
import numpy as np


def conv2_gray(img, kers, verbose=True):
    '''Does a 2D convolution operation on GRAYSCALE `img` using kernels `kers`.
    Uses 'same' boundary conditions.

    Parameters:
    -----------
    img: ndarray. Grayscale input image to be filtered. shape=(height img_y (px), width img_x (px))
    kers: ndarray. Convolution kernels. shape=(Num kers, ker_sz (px), ker_sz (px))
        NOTE: Kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    filteredImg: ndarray. `img` filtered with all the `kers`. shape=(Num kers, img_y, img_x)

    '''
    img_y, img_x = img.shape
    n_kers, ker_y, ker_x = kers.shape

    if verbose:
        print(f'img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, ker_y={ker_y}, ker_x={ker_x}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return
    
    # form kernel / flip it
    kers = kers[:, ::-1, ::-1]

    # padded image
    pad_x = np.ceil((ker_x - 1)/2).astype(int)
    pad_y = np.ceil((ker_y - 1)/2).astype(int)
    padded = np.zeros((img_y + 2*pad_y, img_x + 2*pad_x))
    padded[pad_y:pad_y+img_y, pad_x:pad_x+img_x] = img

    filteredImg = np.zeros((n_kers, img_y, img_x))

    # multiply

    for idx, ker in enumerate(kers):        #loop over kern
        for i in range(img_y):              # over rows
            for j in range(img_x):          # over cols
                patch = padded[i:i+ker_y, j:j+ker_x]
                filteredImg[idx, i, j] = np.sum(ker * patch)

    return filteredImg




def conv2(img, kers, verbose=True):
    '''Does a 2D convolution operation on COLOR or grayscale `img` using kernels `kers`.
    Uses 'same' boundary conditions.

    Parameters:
    -----------
    img: ndarray. Input image to be filtered. shape=(N_CHANS, height img_y (px), width img_x (px))
        where n_chans is 1 for grayscale images and 3 for RGB color images
    kers: ndarray. Convolution kernels. shape=(Num filters, ker_sz (px), ker_sz (px))
        NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    filteredImg: ndarray. `img` filtered with all `kers`. shape=(Num filters, N_CHANS, img_y, img_x)

    -----------
    - N_CHANS, see above.

    '''
    n_chans, img_y, img_x = img.shape
    n_kers, ker_x, ker_y = kers.shape

    if verbose:
        print(f'n_chan={n_chans}, img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, ker_x={ker_x}, ker_y={ker_y}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return
    
    # form kernel / flip it
    kers = kers[:, ::-1, ::-1]

    # padded image
    pad_x = np.ceil((ker_x - 1)/2).astype(int)
    pad_y = np.ceil((ker_y - 1)/2).astype(int)
    padded = np.zeros((n_chans, img_y + 2*pad_y, img_x + 2*pad_x))
    padded[:, pad_y:pad_y+img_y, pad_x:pad_x+img_x] = img



    filteredImg = np.zeros((n_kers,n_chans, img_y, img_x))

    # multiply

    for idx, ker in enumerate(kers):        #loop over kern
        for i in range(img_y):              # over rows
            for j in range(img_x):          # over cols
                patch = padded[:, i:i+ker_y, j:j+ker_x]
                filteredImg[idx, :, i, j] = np.sum(ker * patch, axis=(1,2))

    return filteredImg


def conv2nn(imgs, kers, bias, verbose=True):
    '''General 2D convolution operation suitable for a convolutional layer of a neural network.
    Uses 'same' boundary conditions.

    Parameters:
    -----------
    imgs: ndarray. Input IMAGES to be filtered. shape=(BATCH_SZ, n_chans, img_y, img_x)
        where batch_sz is the number of images in the mini-batch
        n_chans is 1 for grayscale images and 3 for RGB color images
    kers: ndarray. Convolution kernels. shape=(n_kers, N_CHANS, ker_sz, ker_sz)
        NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    bias: ndarray. Bias term used in the neural network layer. Shape=(n_kers,)
        i.e. there is a single bias per filter. Convolution by the c-th filter gets the c-th
        bias term added to it.
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    What's new (vs conv2):
    -----------
    - Multiple images (mini-batch support)
    - Kernels now have a color channel dimension too
    - Collapse (sum) over color channels when computing the returned output images
    - A bias term

    Returns:
    -----------
    output: ndarray. `imgs` filtered with all `kers`. shape=(BATCH_SZ, n_kers, img_y, img_x)


    '''
    batch_sz, n_chans, img_y, img_x = imgs.shape
    n_kers, n_chans, ker_x, ker_y = kers.shape

    if verbose:
        print(f'n_chan={n_chans}, img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, ker_x={ker_x}, ker_y={ker_y}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return
    
    # form kernel / flip it
    kers = kers[:, :, ::-1, ::-1]

    # padded image
    pad_x = np.ceil((ker_x - 1)/2).astype(int)
    pad_y = np.ceil((ker_y - 1)/2).astype(int)
    padded = np.zeros((batch_sz, n_chans, img_y + 2*pad_y, img_x + 2*pad_x))
    padded[:, :, pad_y:pad_y+img_y, pad_x:pad_x+img_x] = imgs

    filteredImgs = np.zeros((batch_sz, n_kers, img_y, img_x))

    # multiply

    for idx, ker in enumerate(kers):     #loop over kern
        for i in range(img_y):              # over rows
            for j in range(img_x):          # over cols
                patch = padded[:, :, i:i+ker_y, j:j+ker_x]
                filteredImgs[:, idx, i, j] = np.sum(ker * patch, axis=(1,2,3)) + bias[idx]

    return filteredImgs


def get_pooling_out_shape(img_dim, pool_size, strides):
    '''Computes the size of the output of a max pooling operation along one spatial dimension.

    Parameters:
    -----------
    img_dim: int. Either img_y or img_x
    pool_size: int. Size of pooling window in one dimension: either x or y (assumed the same).
    strides: int. Size of stride when the max pooling window moves from one position to another.

    Returns:
    -----------
    int. The size in pixels of the output of the image after max pooling is applied, in the dimension
        img_dim.
    '''
    pool_size = np.floor((img_dim - pool_size)/strides) + 1
    return pool_size.astype("uint8")


def max_pool(inputs, pool_size=2, strides=1, verbose=True):
    ''' Does max pooling on inputs. Works on single grayscale images, so somewhat comparable to
    `conv2_gray`.

    Parameters:
    -----------
    inputs: Input to be filtered. shape=(height img_y, width img_x)
    pool_size: int. Pooling window extent in both x and y.
    strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    outputs: Input filtered with max pooling op. shape=(out_y, out_x)
        NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
        not preserved (unless pool_size=1...but what's the point of that? :)

    NOTE: There is no padding in the max-pooling operation.

    '''
    img_y, img_x = inputs.shape

    out_y = get_pooling_out_shape(img_y, pool_size, strides)
    out_x = get_pooling_out_shape(img_x, pool_size, strides)

    # form the output array
    out_img = np.zeros((out_y, out_x))

    img_idx_x = 0
    img_idx_y = 0
    
    # max pool
    for i in range(out_y):  # over rows
        for j in range(out_x):  # over cols
            # pool window
            window = inputs[img_idx_y:img_idx_y + pool_size, img_idx_x:img_idx_x + pool_size]
            out_img[i,j] = np.max(window)
            img_idx_x += strides
        img_idx_x = 0
        img_idx_y += strides

    return out_img



def max_poolnn(inputs, pool_size=2, strides=1, verbose=True):
    ''' Max pooling implementation for a MaxPool2D layer of a neural network

    Parameters:
    -----------
    inputs: Input to be filtered. shape=(mini_batch_sz, n_chans, height img_y, width img_x)
        where n_chans is 1 for grayscale images and 3 for RGB color images
    pool_size: int. Pooling window extent in both x and y.
    strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    outputs: Input filtered with max pooling op. shape=(mini_batch_sz, n_chans, out_y, out_x)
        NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
        not preserved (unless pool_size=1...but what's the point of that?)

    What's new (vs max_pool):
    -----------
    - Multiple images (mini-batch support)
    - Images now have a color channel dimension too

    '''
    mini_batch_sz, n_chans, img_y, img_x = inputs.shape

    out_y = get_pooling_out_shape(img_y, pool_size, strides)
    out_x = get_pooling_out_shape(img_x, pool_size, strides)

    # form the output array
    out_img = np.zeros((mini_batch_sz, n_chans, out_y, out_x))

    img_idx_x = 0
    img_idx_y = 0
    
    # max pool
    for i in range(out_y):  # over rows
        for j in range(out_x):  # over cols
            # pool window
            window = inputs[:, :, img_idx_y:img_idx_y + pool_size, img_idx_x:img_idx_x + pool_size]
            out_img[:,:,i,j] = np.max(window, axis=(2,3))
            img_idx_x += strides
        img_idx_x = 0
        img_idx_y += strides

    return out_img