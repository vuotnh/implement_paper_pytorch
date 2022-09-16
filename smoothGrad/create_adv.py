import torch
import cv2
from torch.autograd import Variable
import numpy as np
from gradients import SmoothGrad, VanillaGrad
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from torchvision.models import vgg19
from image_utils import preprocess_image, save_as_gray_image
from labels import IMAGENET_LABELS
import matplotlib.pyplot as plt
import ImageNet_label
def fgsm_attack(image, epsilon, data_grad):
    data_grad = data_grad.reshape([224, 224, 3])
    sign = data_grad.sign()
    adv_image = image + epsilon*sign.detach().numpy()
    adv_image = np.clip(adv_image, 0, 1)
    return adv_image

def main():
    epsilon = 0.2
    image_path = "./data/0000802.jpg"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    images = preprocess_image(img) 
    model = vgg19(pretrained=True)
    
    # prediction
    output = model(images)
    pred_index = np.argmax(output.data.cpu().numpy())
    target_label = Variable(torch.LongTensor([pred_index]), requires_grad=False)
    print(f"Prediction label: {ImageNet_label.IMAGENET_LABEL[pred_index]}")

    images.requires_grad = True
    images.retain_grad()

    output = model(images)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output, target_label)
    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    adv_image = fgsm_attack(img, epsilon, data_grad)

    # prediction adv
    output_adv = model(preprocess_image(adv_image))
    pred_index_adv = np.argmax(output_adv.data.cpu().numpy())
    print(f"Prediction label adv: {ImageNet_label.IMAGENET_LABEL[pred_index_adv]}")
    cv2.imwrite("adv.jpg",adv_image)

if __name__ == '__main__':
    main()