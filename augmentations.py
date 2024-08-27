import albumentations as A

def get_augmentations():
    return A.Compose([
        A.Resize(224, 224)
    ])
