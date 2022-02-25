import albumentations as A
import numpy as np
import matplotlib.pyplot as plt

# Verileri göstermek için yardımcı bir fonksiyon.
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    

# Verileri doğru bir şekilde göstermek için 0-1 aralığına çevirir.   
def denormalize(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    

# Modelin bulduğu maskeyi kırpar.
def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

# Maske resmini sadece 1 veya 0'a çevirir.
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# Training için kullanılan augmentation teknikleri.(albumentations Kütüphanesi kullanarak kurulmuştur)
def get_training_augmentation():
    train_transform = [
        A.Resize(512,512,always_apply=True),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1) 
    ]
    return A.Compose(train_transform)

# Validasyon için kullanılan augmentation teknikleri.(albumentations Kütüphanesi kullanarak kurulmuştur)
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(512,512,always_apply=True),      
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(test_transform)


# segmentation_models Kütüphanenin seçilen modele göre prepocessing işlemlerini yapar.(Train ederken giriş verisine kullanılmıştır)
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

