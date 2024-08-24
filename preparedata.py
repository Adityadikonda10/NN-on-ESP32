import numpy as np
import struct
import gzip

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Read labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load MNIST test images and labels
images = load_mnist_images('t10k-images-idx3-ubyte.gz')
labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

# Save images as raw binary file for C++ testing
images.tofile('mnist_images.raw')
