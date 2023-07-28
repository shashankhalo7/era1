from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

class Learner():
    def __init__(self,train_loader,test_loader, model, loss_func=F.nll_loss,metrics=None, model_dir='models',reg=(0,0),device='cuda'):
        self.train_loader,self.test_loader,self.model = train_loader,test_loader,model
        self.train_losses,self.train_acc=[],[]
        self.test_losses,self.test_acc=[],[]
        self.loss_func=loss_func
        self.lambda_l1,self.weight_decay=reg
        self.device=device

    def train(self, optimizer, epoch,lambda_l1=0):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct=0
        processed=0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            y_pred = self.model(data)
            loss = self.loss_func(y_pred, target)
            l1=0
            for p in self.model.parameters():
                l1 = l1 +p.abs().sum()
            loss= loss +self.lambda_l1*l1
            self.train_losses.append(loss)
            loss.backward()
            optimizer.step()
            pred = y_pred.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
        return self.train_losses,self.train_acc

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_func(output, target, reduction='sum').item()  
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)


        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        self.test_acc.append(100. * correct / len(self.test_loader.dataset))

        return self.test_losses,self.test_acc

    def fit(self,epochs=10,lr=0.01,step_size=4,gamma=0.1):
        self.model= self.model.to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        EPOCHS = epochs
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch)
            self.train(optimizer, epoch,lambda_l1=self.lambda_l1)
            scheduler.step()
            self.test()

    def fit_custom(self,epochs,optimizer,lr_scheduler,pass_loss=False):
        self.model= self.model.to(self.device)
        optimizer = optimizer
        scheduler = lr_scheduler
        EPOCHS = epochs
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch)
            self.train(optimizer, epoch,lambda_l1=self.lambda_l1)
            self.test()
            if pass_loss:
                scheduler.step(self.test_losses[-1])    
            else:
                scheduler.step()
            


    def summary(self,input_size=None):
      if input_size==None:
        input_size= self.test_loader.image_size
      self.model = self.model.to(self.device)
      summary(self.model, input_size=input_size)

    def plot_losses(self):
      fig, axs = plt.subplots(2,2,figsize=(15,10))
      axs[0, 0].plot(self.train_losses)
      axs[0, 0].set_title("Training Loss")
      axs[1,0].axis(ymin=50,ymax=100)
      axs[1, 0].plot(self.train_acc)
      axs[1, 0].set_title("Training Accuracy")
      axs[0, 1].plot(self.test_losses)
      axs[0, 1].set_title("Test Loss")
      axs[1,1].axis(ymin=50,ymax=100)
      axs[1, 1].plot(self.test_acc)
      axs[1, 1].set_title("Test Accuracy")
    
    
    def predict(self):
        pass

    def save_model(self,PATH="."):
        torch.save(self.model, PATH)

    def load_model(self, file):
      self.model = torch.load(file)
