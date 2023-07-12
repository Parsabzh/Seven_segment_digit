# Seven Segment Number Detection

This project focuses on training a deep learning model to detect and classify numbers represented in the form of seven-segment displays. The model is built using TensorFlow and Keras libraries and leverages convolutional neural networks (CNNs) for accurate classification.

## Dataset

The dataset used in this project is a collection of images of numbers represented in seven-segment displays. The dataset is divided into two sets: a training set and a validation set. The images are preprocessed and resized to a consistent size of 28x28 pixels.

## Model Architecture

The model architecture consists of several layers including convolutional layers, pooling layers, dropout layers, and dense layers. The model is designed to learn features from the input images and make predictions based on the learned features. Dropout is used to prevent overfitting and improve generalization.

## Training and Evaluation

The model is trained using the training set and evaluated using the validation set. The training process is executed for a specified number of epochs, and the accuracy and loss metrics are calculated for each epoch. The trained model is then saved for future use.

## Results

The accuracy and loss metrics are plotted and visualized using Matplotlib to observe the model's performance during training. The graphs show the training accuracy and loss as well as the validation accuracy and loss over the epochs.

## Usage

To run the project, follow these steps:

1. Install the required dependencies specified in a `requirements.txt` file.
2. Prepare your dataset by organizing the images of numbers represented in seven-segment displays into appropriate directories: `train` and `validation`.
3. Update the `train_set` and `test_set` variables in the code to specify the correct paths to the dataset directories.
4. Execute the Python script to train the model and generate the accuracy and loss graphs.
5. The trained model will be saved as `model_saved.h5`.

Feel free to experiment with the code, adjust hyperparameters, and customize the model architecture as per your requirements.

## Dependencies

The project requires the following dependencies:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL

## Credits

This project is developed by Parsabzh.

