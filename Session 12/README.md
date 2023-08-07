# Assignment Overview: Integrating S10 with Lightning, Gradio, and Spaces

## Objective:

The purpose of this assignment is to move the S10 assignment to the Lightning framework and subsequently integrate it with Gradio for user interactivity and Hugging Face Spaces for deployment.

## Steps:

### 1. Migrate S10 to Lightning:
- Retrain the model using PyTorch Lightning. This involves replacing the traditional PyTorch training loop with the LightningModule.
- Ensure all required callbacks, metrics, and other features of Lightning are correctly integrated.

### 2. Gradio Integration:
- Create a Gradio interface that wraps around the model and allows for interactive predictions.
- Add user interactivity features:
  - **GradCAM Options**:
    - Ask users whether they want to see GradCAM images.
    - Allow users to select how many GradCAM images they want to view.
    - Allow users to choose from which layer they wish to see the GradCAM visualizations.
    - Provide an opacity change feature for the GradCAM images.
  - **Misclassified Images**:
    - Ask users if they want to view misclassified images.
    - Allow users to select the number of misclassified images they wish to view.
  - **Image Upload**:
    - Allow users to upload new images for predictions.
    - Provide 10 example images for users to play around with.
  - **Class Options**:
    - Allow users to specify how many top classes they want to be shown as predictions.
    - Ensure users can't select more than 10 classes.

### 3. Deployment on Hugging Face Spaces:
- Deploy the Gradio application with the integrated model to Hugging Face Spaces.
- **Spaces README**:
  - Include comprehensive details about what the Spaces app does.
  - Ensure the README on Spaces doesn't contain any training code.
  - Provide links to the Lightning codebase on GitHub, making sure the actual model training details are kept separate from the deployment.

### 4. GitHub:
- Store the Lightning training code separately on a GitHub repository.
- Create a detailed README for this repository, which should include:
  - A log of the training process.
  - Graphs showing the loss function over training epochs.
  - 10 examples of misclassified images, along with their true and predicted labels.

## Deliverables:
1. **Spaces App Link**: A link to the Gradio app deployed on Hugging Face Spaces.
2. **Spaces README Link**: A direct link to the README file within the Spaces deployment, explaining the features and usage of the app.
3. **GitHub Link**: A link to the GitHub repository that contains the PyTorch Lightning training code. This should have its own comprehensive README detailing the training process, loss function graphs, and misclassified images.

---

