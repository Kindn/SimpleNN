import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

if __name__ == "__main__":
    print(os.getcwd())
    images_train, labels_train = load_mnist(".", 
                                            kind='train')
    images_test, labels_test = load_mnist(".", 
                                            kind='t10k')
    #print(images.shape)
    #print(labels.shape)
    images_train = images_train.T
    images_test = images_test.T
    labels_train = labels_train.reshape(1, -1)
    labels_test = labels_test.reshape(1, -1)

    print(labels_train.shape)
    print(labels_test.shape)

    np.savetxt("./images_train.csv", 
                images_train, "%f")
    np.savetxt("./labels_train.csv", 
                labels_train, "%d")
    np.savetxt("./images_test.csv", 
                images_test, "%f")
    np.savetxt("./labels_test.csv", 
                labels_test, "%d")

