Session 7

Shashank Pathak

### Code1.ipynb

**Target**:

1. Get the basic code setup (train/test loops) and dataloader. 
2. Use basic transforms (ToTensor, Normalize)
3. 15 epochs

**Results**:

* Parameters:  194,884
* Best Train Accuracy: 99.03
* Best Test Accuracy: 98.78

**Analysis**:

* The model is quite large for MNIST task 
* Overfitting

### Code2.ipynb

**Target**:

1. Decrease the number of parameters (Decrease Kernels, Add GAP Layer)
2. Add regularisation(Dropout Layers)

**Results**:

* Parameters: 6,950
* Best Train Accuracy: 98.86 %
* Best Test Accuracy: 99.25%

**Analysis**:

* The model performs decently. Potential to improve
* Underfitting


### Code3.ipynb

**Target**:

1. As the previous model was undefitting increase the capacity 

**Results**:

* Parameters: 9,060
* Best Train Accuracy: 98.81
* Best Test Accuracy: 99.39

**Analysis**:

* Capcity can be increased further
* Transforms can be applied to make train set more like test set



### Code5.ipynb

**Target**:

1. Add transforms and increase capacity further, Add StepLR Scheduler

**Results**:

* Parameters: 7,928
* Best Train Accuracy: 98.81
* Best Test Accuracy: 99.44

**Analysis**:

* The can be trained for more epochs, has potential
* Can add OCP for training
