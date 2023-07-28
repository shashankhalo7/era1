import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from gradcam import GradCam


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Interpreter:
    def __init__(self, model, dataloader, classes):
        self.model = model
        self.dataloader = dataloader
        self.classes = classes
        self.pred = []
        self.y = []
        self.incorrect_examples = []
        self.incorrect_labels = []
        self.correct_labels = []

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                self.pred.extend(predicted.cpu())
                self.y.extend(labels.cpu())
                idxs_mask = ((predicted == labels) == False).nonzero()
                self.incorrect_examples.extend(images[idxs_mask].cpu())
                self.incorrect_labels.extend(predicted[idxs_mask].cpu())
                self.correct_labels.extend(labels[idxs_mask].cpu())

        self.y_label = [classes[item] for item in self.y]
        self.pred_label = [classes[item] for item in self.pred]
        self.cm = confusion_matrix(self.y_label, self.pred_label, classes)
        self.accuracy = np.trace(self.cm) / np.sum(self.cm).astype('float')
        self.misclass = 1 - self.accuracy

    def plot_confusion_matrix(self, title='Confusion matrix', cmap=None, normalize=False):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions
        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        cm = self.cm
        target_names = self.classes

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(self.accuracy, self.misclass))
        plt.show()

    def show_classification_report(self):
        report = pd.DataFrame.from_dict(classification_report(self.y_label, self.pred_label, target_names=self.classes, output_dict=True)).T
        return report

    def show_misclassifications(self, k=25, gradcam=False):

        if gradcam:
            fig = plt.figure(figsize=(20, 6 * k))

            for idx in np.arange(k):
                img = self.incorrect_examples[idx][0]
                gm = GradCam(model=self.model, img_tensor=img,
                             correct_class=self.correct_labels[idx][0], classes=self.classes,
                             feature_module=self.model.layer4, target_layer_names=['1'])
                input_img, cam_img = gm.get()
                ax = fig.add_subplot(3 * k, 5, idx + 1)
                ax.set_title(f'pred: {self.classes[self.incorrect_labels[idx]]}'
                             f' / correct: {self.classes[self.correct_labels[idx]]}')
                plt.imshow(cam_img)

            fig.tight_layout()
        else:
            fig = plt.figure(figsize=(20, 6 * k))

            for idx in np.arange(k):
                ax = fig.add_subplot(3 * k, 5, idx + 1)
                ax.set_title(f'pred: {self.classes[self.incorrect_labels[idx]]}'
                             f' / correct: {self.classes[self.correct_labels[idx]]}')
                img = self.incorrect_examples[idx][0]
                imshow(img)
            fig.tight_layout()


