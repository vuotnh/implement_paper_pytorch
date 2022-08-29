import numpy as np
import cv2
import torch
from torch.autograd import Variable

class VanillaGrad:
    def __init__(self, pretrained_model, cuda=False) -> None:
        self.pretrained_model = pretrained_model
        self.features = pretrained_model.features
        self.cuda = cuda

    def __call__(self, input_image, index=None):
        output = self.pretrained_model(input_image)
        if index is not None:
            index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        if self.cuda:
            one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
        else:
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot * output)

        # khi train model, cần train nhiều epochs, mỗi epochs lại gồm nhiều step
        # do đó nên mk cần backward nhiều lần để tính đạo hàm ngược lại
        # để tính backward nhiều lần ta cần thuộc tính retain_graph=True
        # Tuy nhiên khi thực hiện backward nhiều lần thì đạo hàm sẽ được cộng dồn vào leaf tensor
        # để tránh cộng dồn thì ở mỗi step dùng gradient descent xong thì mk thường zero_grad trước khi sang step khác
        one_hot.backward(retain_graph=True)

        grad = input_image.grad.data.cpu().numpy()
        grad = grad[0, :, :, :]

        return grad


class SmoothGrad(VanillaGrad):

    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15,
                 n_samples=25, magnitude=True):
        super(SmoothGrad, self).__init__(pretrained_model, cuda)
        """
        self.pretrained_model = pretrained_model
        self.features = pretrained_model.features
        self.cuda = cuda
        self.pretrained_model.eval()

        @param: stdev_spread : standard deviation for noise 
        @param: n_samples: number of neighborhood pixel
        """
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude

    def __call__(self, input_image, index=None):
        input_image = input_image.data.cpu().numpy()
        stdev = self.stdev_spread * (np.max(input_image) - np.min(input_image))
        total_gradients = np.zeros_like(input_image)
        for i in range(self.n_samples):
            noise = np.random.normal(0, stdev, input_image.shape).astype(np.float32)
            x_plus_noise = input_image + noise
            if self.cuda:
                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).cuda(), requires_grad=True)
            else:
                x_plus_noise = Variable(torch.from_numpy(x_plus_noise), requires_grad=True)
            output = self.pretrained_model(x_plus_noise)

            if index is None:
                index = np.argmax(output.data.cpu().numpy())

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            if self.cuda:
                one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
            else:
                one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            one_hot = torch.sum(one_hot * output)

            if x_plus_noise.grad is not None:
                x_plus_noise.grad.data.zero_()
            one_hot.backward(retain_graph=True)

            grad = x_plus_noise.grad.data.cpu().numpy()

            if self.magnitutde:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad
            #if self.visdom:

        avg_gradients = total_gradients[0, :, :, :] / self.n_samples

        return avg_gradients


class GuidedBackpropReLU(torch.autograd.Function):

    def __init__(self, inplace=False):
        super(GuidedBackpropReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        pos_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            input,
            pos_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors

        pos_mask_1 = (input > 0).type_as(grad_output)
        pos_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, pos_mask_1),
                pos_mask_2)

        return grad_input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class GuidedBackpropGrad(VanillaGrad):

    def __init__(self, pretrained_model, cuda=False):
        super(GuidedBackpropGrad, self).__init__(pretrained_model, cuda)
        for idx, module in self.features._modules.items():
            if module.__class__.__name__ is 'ReLU':
                self.features._modules[idx] = GuidedBackpropReLU()


class GuidedBackpropSmoothGrad(SmoothGrad):

    def __init__(self, pretrained_model, cuda=False, stdev_spread=.15, n_samples=25, magnitude=True):
        super(GuidedBackpropSmoothGrad, self).__init__(
            pretrained_model, cuda, stdev_spread, n_samples, magnitude)
        for idx, module in self.features._modules.items():
            if module.__class__.__name__ is 'ReLU':
                self.features._modules[idx] = GuidedBackpropReLU()

