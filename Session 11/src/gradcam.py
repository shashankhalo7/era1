import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class GradCam():
    def __init__(self, model, img_tensor, correct_class, classes, feature_module, target_layer_names):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.img_tensor = img_tensor.unsqueeze_(0)
        self.classes = classes
        self.correct_class = self.classes[correct_class]
        self.model = model

        target_activations = []
        x = self.img_tensor.to(device)
        for name, module in model._modules.items():
            if module == model.layer4:
                target_activations, x = self.extract_features(x, feature_module, target_layer_names)
            elif "linear" in name.lower():
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)
                x = module(x)
            else:
                x = module(x)

        features, output = target_activations, x
        index = np.argmax(output.cpu().data.numpy())
        self.pred_class = self.classes[index]
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.layer4.zero_grad()
        model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.gradients[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, self.img_tensor.shape[2:])
        cam = cam - np.min(cam)
        self.cam = cam / np.max(cam)

    def extract_features(self, input, model, target_layers):
        x = input
        outputs = []
        self.gradients = []
        for name, module in model._modules.items():
            x = module(x)
            if name in target_layers:
                x.register_hook(lambda grad: self.gradients.append(grad))
                outputs += [x]
        return outputs, x

    def plot(self):
        heatmap = cv2.applyColorMap(np.uint8(255 * self.cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img = self.img_tensor[0] / 2 + 0.5
        img = np.transpose(img.numpy(), (1, 2, 0))
        cam = heatmap + img
        cam = cam / np.max(cam)
        fig = plt.figure(figsize=(20, 10))

        ax = fig.add_subplot(3, 5, 1)
        ax.set_title('Input Image')
        plt.imshow(img)

        ax = fig.add_subplot(3, 5, 2)
        ax.set_title(f'pred: {self.pred_class}'
                     f' / correct: {self.correct_class}')
        plt.imshow(cam)

        fig.tight_layout()

    def get(self):
        heatmap = cv2.applyColorMap(np.uint8(255 * self.cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img = self.img_tensor[0] / 2 + 0.5
        img = np.transpose(img.numpy(), (1, 2, 0))
        cam = heatmap + img
        cam = cam / np.max(cam)
        return img, cam