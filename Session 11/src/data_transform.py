import albumentations as A
from albumentations.pytorch import ToTensorV2


class MNISTTransforms:
    def __init__(self):
        pass

    def build_transforms(self, train_tfms_list=[], test_tfms_list=[]):
        train_tfms_list.extend([A.Normalize(mean=[0.1307], std=[0.3081]), ToTensorV2()])
        test_tfms_list.extend([A.Normalize(mean=[0.1307], std=[0.3081]), ToTensorV2()])
        return A.Compose(train_tfms_list), A.Compose(test_tfms_list)

class CIFAR10Transforms:
    def __init__(self):
        pass

    def build_transforms(self,  train_tfms_list=[], test_tfms_list=[]):
        train_tfms_list.extend([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()])
        test_tfms_list.extend([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()])
        return A.Compose(train_tfms_list), A.Compose(test_tfms_list)
