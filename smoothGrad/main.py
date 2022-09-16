import cv2
import torch
from torch.autograd import Variable
from torchvision.models import vgg19
import numpy as np
from gradients import SmoothGrad, VanillaGrad

from image_utils import preprocess_image, save_as_gray_image
from labels import IMAGENET_LABELS


def main():
    image_path = "adv.jpg"
    img = cv2.imread(image_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    preprocessed_image = preprocess_image(img) 
    model = vgg19(pretrained=True)

    # prediction
    output = model(preprocessed_image)
    pred_index = np.argmax(output.data.cpu().numpy())
    print(f"Prediction label: {IMAGENET_LABELS[pred_index]}")

    # vanilla gradient
    vanilla_grad = VanillaGrad(pretrained_model=model)
    vanilla_saliency = vanilla_grad(preprocessed_image)
    save_as_gray_image(vanilla_saliency, 'vanilla_grad.jpg')

    # smoothgrad
    smooth_grad = SmoothGrad(pretrained_model=model)
    smooth_saliency = smooth_grad(preprocessed_image)
    save_as_gray_image(smooth_saliency, 'smooth_grad.jpg')

if __name__ == '__main__':
    main()