from PIL import Image
from torchvision import transforms as tr
from albumentations import *
from albumentations.pytorch import ToTensorV2

def trans1():
    tr1 = tr.Compose([tr.ColorJitter(contrast=20),
                      tr.RandomHorizontalFlip()])
    return tr1


def trans2():
    tr1 = Compose([ColorJitter(brightness=0, contrast=20, always_apply=False, p=0.5),
                   HorizontalFlip(always_apply=False, p=0.5),
                   HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False,
                                      p=0.5),
                   OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None,
                                     mask_value=None, always_apply=False, p=0.5),
                   ShiftScaleRotate(scale_limit=0.3, interpolation=1, border_mode=4, p=0.5),
                   Resize(height=224, width=224),
                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                   ToTensorV2(),
                   ])

    return tr1


def test01(img_path):
    tr = trans1()
    img = Image.open(img_path)
    img = tr(img)
    img.show()


def test02(img_path):
    tr = trans2()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tr(image=img)['image']
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    img_path = "D:\PythonProjects\Pee\dataset\HYAL/01.PNG"
    test02(img_path)

