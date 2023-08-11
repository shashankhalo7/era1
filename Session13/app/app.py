import lightning.pytorch as pl
import torch.nn as nn
import gradio as gr
from torchvision import transforms
import itertools
from src.models_lt import *
from src.utils import *

import os
import random
from PIL import Image
import numpy as np

def read_images(directory, n):
    files = os.listdir(directory)
    random_files = random.sample(files, n)
    image_list = []
    for file in random_files:
        if file.endswith('.jpg') or file.endswith('.png'):  # add more conditions if there are other image types
            image = Image.open(os.path.join(directory, file))
            image_array = np.array(image)
            image_list.append(image_array)
    return image_list



transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def inference(image,gradcam,num_gradcam,opacity,layer,misclassified,num_misclassified,topk):
    model =  CustomResnet()
    model.load_state_dict(torch.load("cifar10_model.pth",map_location=torch.device('cpu')), strict=False)
    softmax = nn.Softmax(dim=1)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    input = transform(image)
    input = input.unsqueeze(0)
    output = model(input)
    probs = softmax(output).flatten()
    confidences = {class_names[i]: float(probs[i]) for i in range(10)}
    sorted_confidences = dict(sorted(confidences.items(), key=lambda item: item[1],reverse = True))
    confidence_score = dict(itertools.islice(sorted_confidences.items(), topk))
    pred = probs.argmax(dim=0, keepdim=True)
    pred = pred.item()
    if gradcam == 'Yes':
      target_layers = [model.res_block3[3*layer]]
      cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
      image = input.cpu().numpy()
      grayscale_cam = cam(input_tensor=input, targets=[ClassifierOutputTarget(pred)],aug_smooth=True,eigen_smooth=True)
      grayscale_cam = grayscale_cam[0, :]
      visualization = show_cam_on_image(imshow(image.squeeze(0)), grayscale_cam, use_rgb=True,image_weight=opacity)
      print(imshow(image.squeeze(0)).shape)
      if misclassified == 'Yes':
        misclass = read_images('outputs/misclass', n=num_misclassified)
        cam = read_images('outputs/cam', n=num_gradcam)
        return confidence_score,visualization,misclass,cam
      else:
        cam = read_images('outputs/cam', n=num_gradcam)
        return confidence_score,visualization,None,cam
    else:
      if misclassified == 'Yes':
        misclass = read_images('outputs/misclass', n=num_misclassified)
        return confidence_score,None,misclass,None
      else:
        return confidence_score,None,None,None

with gr.Blocks() as demo:
  with gr.Row() as interface:
    with gr.Column() as input_panel:
      image = gr.Image(shape=(32,32))

      gradcam = gr.Radio(label="Do you Need GradCam Output", choices=["Yes", "No"])

      with gr.Column(visible=False) as gradcam_details:
          num_gradcam = gr.Slider(minimum = 0, maximum=20, value = 0,step=1, label="Number of Gradcam Images")
          opacity = gr.Slider(minimum = 0, maximum=1, value = 0.5,step=0.1, label="Opacity of image overlayed by gradcam output")
          layer = gr.Slider(minimum = -2, maximum=-1, value = -1,step=1, label="Which layer?")

      def filter_gradcam(gradcam):
        if gradcam == 'Yes':
          return gr.update(visible=True)
        else:
          return gr.update(visible=False)
        
      gradcam.change(filter_gradcam, gradcam, gradcam_details)

      misclassified = gr.Radio(label="Do you see misclassified Images", choices=["Yes", "No"])

      with gr.Column(visible=False) as misclassified_details:
          num_misclassified = gr.Slider(minimum = 0, maximum=20, value = 0, label="Number of Misclassified Images")

      def filter_misclassified(misclassified):
        if misclassified == 'Yes':
          return gr.update(visible=True)
        else:
          return gr.update(visible=False)
        
      misclassified.change(filter_misclassified, misclassified, misclassified_details)

      topk = gr.Slider(minimum = 1, maximum=10, value = 1, step=1, label="Number of Classes")
      btn = gr.Button("Classify")

    with gr.Column() as output_panel:
      gradcam_output = gr.Image(shape=(32, 32), label="Output").style(height=240, width=240)
      output_labels = gr.Label(num_top_classes=10)
      misclassified_gallery = gr.Gallery(label="Misclassified Images")
      gradcam_gallery = gr.Gallery(label="Some More GradCam Outputs")


  
  btn.click(fn=inference, inputs=[image,gradcam,num_gradcam,opacity,layer,misclassified,num_misclassified,topk], outputs=[output_labels,gradcam_output,misclassified_gallery,gradcam_gallery])

  gr.Examples(
        [
            ['examples/automobile1.jpg', 'Yes', 12, 0.8, -2, 'Yes', 10, 1],
            ['examples/airplane1.jpg', 'No', 2, 0.8, -1, 'No', 10, 5],
            ['examples/deer2.jpg', 'No', 4, 0.5, -1, 'Yes', 12, 2],
            ['examples/truck1.jpg', 'No', 17, 0.9, -1, 'No', 18, 4],
            ['examples/deer2.jpg', 'Yes', 7, 1.0, -1, 'No', 10, 5],
            ['examples/dog2.jpg', 'Yes', 16, 0.3, -2, 'Yes', 12, 9],
            ['examples/cat1.jpg', 'No', 14, 0.9, -2, 'No', 10, 6],
            ['examples/cat1.jpg', 'Yes', 20, 0.9, -1, 'Yes', 13, 3],
            ['examples/deer1.jpg', 'Yes', 11, 0.7, -1, 'Yes', 7, 4],
            ['examples/bird1.jpg', 'Yes', 14, 0.2, -2, 'Yes', 17, 5],
            ['examples/automobile2.jpg', 'Yes', 2, 0.7, -2, 'Yes', 15, 10],
            ['examples/truck1.jpg', 'Yes', 13, 0.8, -1, 'Yes', 18, 9],
            ['examples/deer2.jpg', 'Yes', 11, 0.0, -1, 'Yes', 18, 4],
            ['examples/cat2.jpg', 'Yes', 10, 0.3, -2, 'No', 1, 10],
            ['examples/automobile2.jpg', 'Yes', 2, 0.1, -1, 'No', 15, 6],
            ['examples/horse2.jpg', 'Yes', 7, 0.5, -2, 'Yes', 2, 8],
            ['examples/bird1.jpg', 'No', 15, 0.3, -2, 'Yes', 4, 10],
            ['examples/truck2.jpg', 'Yes', 15, 1.0, -2, 'Yes', 6, 9],
            ['examples/airplane1.jpg', 'No', 19, 0.5, -2, 'Yes', 20, 7],
            ['examples/ship2.jpg', 'Yes', 12, 0.5, -1, 'Yes', 6, 2],
            ['examples/airplane1.jpg', 'Yes', 3, 0.6, -1, 'No', 18, 6],
            ['examples/truck2.jpg', 'No', 14, 0.7, -2, 'No', 8, 1],
            ['examples/frog1.jpg', 'Yes', 1, 0.5, -2, 'No', 1, 2],
            ['examples/automobile2.jpg', 'Yes', 9, 0.5, -1, 'No', 2, 8],
            ['examples/deer2.jpg', 'Yes', 9, 0.0, -2, 'No', 14, 5],
            ['examples/bird2.jpg', 'No', 14, 0.6, -2, 'Yes', 18, 2],
            ['examples/frog2.jpg', 'No', 10, 0.8, -2, 'No', 13, 4],
            ['examples/cat1.jpg', 'Yes', 10, 0.5, -1, 'No', 7, 2],
            ['examples/bird2.jpg', 'Yes', 3, 0.5, -2, 'Yes', 17, 1],
            ['examples/cat2.jpg', 'Yes', 5, 0.9, -2, 'No', 19, 8],
            ['examples/ship2.jpg', 'Yes', 20, 0.7, -2, 'Yes', 10, 4],
            ['examples/deer2.jpg', 'Yes', 10, 0.4, -2, 'No', 9, 3],
            ['examples/dog2.jpg', 'No', 17, 0.8, -1, 'Yes', 3, 9],
            ['examples/cat1.jpg', 'Yes', 20, 0.7, -1, 'No', 8, 8],
            ['examples/automobile1.jpg', 'Yes', 4, 0.8, -1, 'No', 1, 8],
            ['examples/truck1.jpg', 'No', 7, 1.0, -2, 'No', 2, 2],
            ['examples/dog2.jpg', 'Yes', 5, 0.1, -2, 'Yes', 2, 3],
            ['examples/automobile2.jpg', 'No', 20, 0.2, -2, 'No', 3, 7],
            ['examples/ship2.jpg', 'No', 6, 0.9, -1, 'No', 3, 8],
            ['examples/truck2.jpg', 'No', 1, 0.2, -1, 'Yes', 6, 9],
            ['examples/horse1.jpg', 'No', 1, 0.6, -1, 'Yes', 19, 2],
            ['examples/dog1.jpg', 'No', 1, 0.4, -2, 'No', 18, 8],
            ['examples/cat2.jpg', 'No', 9, 0.5, -1, 'No', 11, 4],
            ['examples/deer2.jpg', 'Yes', 14, 0.2, -2, 'Yes', 11, 10],
            ['examples/ship1.jpg', 'No', 16, 0.6, -1, 'No', 17, 3],
            ['examples/cat1.jpg', 'No', 3, 0.8, -2, 'Yes', 5, 9],
            ['examples/bird2.jpg', 'Yes', 19, 0.3, -1, 'No', 1, 2],
            ['examples/cat1.jpg', 'No', 14, 0.4, -1, 'No', 9, 4],
            ['examples/automobile2.jpg', 'No', 19, 0.3, -1, 'No', 5, 3],
            ['examples/truck1.jpg', 'No', 3, 0.7, -1, 'No', 18, 10]
        ],
        [image,gradcam,num_gradcam,opacity,layer,misclassified,num_misclassified,topk],
        [output_labels,gradcam_output,misclassified_gallery,gradcam_gallery],
        inference,
        cache_examples=True,
    )

if __name__ == "__main__":
    demo.launch(debug=True,share=True)