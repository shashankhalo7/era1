from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from tqdm import tqdm
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# add dynamic way to pull dataset mean std in transforms

def get_train_transforms():
  train_transforms = A.Compose([
    A.PadIfNeeded(min_height=36, min_width=36, always_apply=True, p=1),
    A.RandomCrop(height=32, width=32, always_apply=True, p=1),
    #A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,always_apply=False,fill_value=(0.5, 0.5, 0.5)),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
  ])

  return train_transforms

def get_test_transforms():
  test_transforms = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
  ])
  return test_transforms

def get_augmentation(transforms):
  return lambda img: transforms(image=np.array(img))['image']

def get_cifar_data(train_transforms,test_transforms,bs=512):
  """
  return cifar dataloader object
  """

  train = datasets.CIFAR10('./data', train=True, download=True, transform=get_augmentation(train_transforms))
  test = datasets.CIFAR10('./data', train=False, download=True, transform=get_augmentation(test_transforms))
  
  # For reproducibility
  SEED = 27
  torch.manual_seed(SEED)

  cuda = torch.cuda.is_available()
  if cuda:
    torch.cuda.manual_seed(SEED)

  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=bs, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=bs)

  # train dataloader
  train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

  return train_loader, test_loader
  
def get_model_summary(model,input_shape):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = model.to(device)
  summary(model, input_size=input_shape)

def train(model, device, train_loader, optimizer,scheduler,epoch,criterion):
  train_losses_lst,train_acc_lst = [],[]

  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses_lst.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc_lst.append(100*correct/processed)

  return train_losses_lst,train_acc_lst

def test(model, device, test_loader,criterion):
  test_losses_lst,test_acc_lst = [],[]
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          test_loss += criterion(output, target).item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_losses_lst.append(test_loss)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

  test_acc_lst.append(100. * correct / len(test_loader.dataset))
  return test_losses_lst,test_acc_lst

def plot_metrics(train_losses,test_losses,train_acc,test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot([x.cpu().item() for x in train_losses])
  axs[0, 0].set_title("Training Loss")
  axs[1,0].axis(ymin=0,ymax=100)
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1,1].axis(ymin=0,ymax=100)
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")
  plt.tight_layout()
  plt.show()

def get_incorrect_images(model,test_loader,n=10):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  incorrect_images = []
  predicted_labels = []
  correct_labels = []
  for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    incorrect_items = pred.ne(target.view_as(pred))
    incorrect_indices = incorrect_items.view(-1).nonzero().view(-1)
    predicted_labels.extend([item.item() for item in pred[incorrect_indices[:n-len(incorrect_images)]]])
    correct_labels.extend([item.item() for item in target.view_as(pred)[incorrect_indices[:n-len(incorrect_images)]]])
    incorrect_images.extend([item for item in data[incorrect_indices[:n-len(incorrect_images)]]])
    if len(incorrect_images)==n:
      break
  return incorrect_images,predicted_labels,correct_labels

def imshow(img):
  img = img / 2 + 0.5     # Unnormalize
  npimg = img
  npimg = np.clip(npimg, 0, 1)  # Add this line to clip the values
  return np.transpose(npimg, (1, 2, 0))  # Convert from Tensor image

def plot_misclassified(n,model,test_loader,class_names,cam = True):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  misclas_fig, misclas_axes = plt.subplots(2, 5, figsize=(16, 8))

  incorrect_images,predicted_labels,correct_labels = get_incorrect_images(model,test_loader,n=10)

  for i, image_tensor in enumerate(incorrect_images):
      ax = misclas_axes[i // 5, i % 5]  # Get the location of the subplot
      image = image_tensor.cpu().numpy()
      ax.imshow(imshow(image))  # Display the image
      ax.set_title(f"Predicted {class_names[predicted_labels[i]]}, Actual {class_names[correct_labels[i]]}")  # Set the title as the index

  if cam == True:
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    cam_fig, cam_axes = plt.subplots(2, 5, figsize=(16, 8))

    for i, image_tensor in enumerate(incorrect_images):
        ax = cam_axes[i // 5, i % 5]  # Get the location of the subplot
        image = image_tensor.cpu().numpy()
        grayscale_cam = cam(input_tensor=image_tensor.reshape(1,3,32,32), targets=[ClassifierOutputTarget(predicted_labels[i])],aug_smooth=True,eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(imshow(image), grayscale_cam, use_rgb=True,image_weight=0.6)
        #ax.imshow(np.transpose(imshow(visualization), (2, 0, 1)))  # Display the image
        ax.imshow(visualization,interpolation='bilinear')
        ax.set_title(f"Predicted {class_names[predicted_labels[i]]}, Actual {class_names[correct_labels[i]]}")  # Set the title as the index

  plt.tight_layout()  # To provide sufficient spacing between subplots
  plt.show()







