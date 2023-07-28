DATA_DIR = 'data'

train_albumentations_transform = A.Compose([
    A.RandomCrop(32, 32, p=0.8),
    A.HorizontalFlip(),
    A.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    ToTensorV2()
])

test_albumentations_transform = A.Compose([
    A.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    ToTensorV2()
])

image_loader = ImageDataLoader(train_albumentations_transform, test_albumentations_transform, DATA_DIR, 128, True, 'CIFAR10', figure_size=(20,10))