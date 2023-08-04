import lightning.pytorch as pl
import torch.nn as nn
import gradio as gr
from torchvision import transforms
import itertools
from src.models_lt import *
from src.utils import *


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

model =  CustomResnet()
model.load_state_dict(torch.load("cifar10_model.pth",map_location=torch.device('cpu')), strict=False)
softmax = nn.Softmax(dim=1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



def inference(image,gradcam,num_gradcam,opacity,layer,misclassified,num_misclassified,topk):
    input = transform(image)
    input = input.unsqueeze(0)
    output = model(input)
    probs = softmax(output).flatten()
    confidences = {class_names[i]: float(probs[i]) for i in range(10)}
    sorted_confidences = dict(sorted(confidences.items(), key=lambda item: item[1],reverse = True))
    confidences = dict(itertools.islice(sorted_confidences.items(), topk))
    return confidences,None

with gr.Blocks() as demo:
  with gr.Row() as interface:
    with gr.Column() as input_panel:
      image = gr.Image(shape=(32,32))

      gradcam = gr.Radio(label="Do you Need GradCam Output", choices=["Yes", "No"])

      with gr.Column(visible=False) as gradcam_details:
          num_gradcam = gr.Slider(minimum = 0, maximum=20, value = 0, label="Number of Gradcam Images")
          opacity = gr.Slider(minimum = 0, maximum=1, value = 0.5, label="Opacity of image overlayed by gradcam output")
          layer = gr.Slider(minimum = -2, maximum=-1, value = -1, label="Which layer?")

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
      output_labels = gr.Label(num_top_classes=10)
      gradcam_output = gr.Image(shape=(32, 32), label="Output").style(width=128, height=128)

  
  btn.click(fn=inference, inputs=[image,gradcam,num_gradcam,opacity,layer,misclassified,num_misclassified,topk], outputs=[output_labels,gradcam_output])


if __name__ == "__main__":
    demo.launch(debug=True,share=True)