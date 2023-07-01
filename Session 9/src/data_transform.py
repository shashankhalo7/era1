import torchvision.transforms as T


class MNISTTransforms:
    def __init__(self):
        pass

    def build_transforms(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])


class CIFAR10Transforms:
    def __init__(self):
        pass

    def build_transforms(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
