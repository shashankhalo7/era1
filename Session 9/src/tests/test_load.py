from ..data_loader import data_loader, data_transform

DATA_DIR = '../data'

mnist = data_transform.MNISTTransforms()
mnist_transform = mnist.build_transforms()

train_loader = data_loader.ImageDataLoader(mnist_transform, DATA_DIR, 32, True, 'MNIST')
test_loader = train_loader.test_split()