# Fruit classification project
Fruit Classification is a machine learning project focused on classifying images of 10 different types of fruits using deep learning techniques. The goal is to train a convolutional neural network ResNet-18 to accurately recognize and categorize various fruit types based on their visual characteristics.

Data: https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class/data

## Used Evaluation Metrics
- **Accuracy:** The ratio of correctly classified samples to the total number of samples, indicating the model's overall effectiveness.
- **Precision:** Measures the accuracy in identifying positive examples, calculated as true positives divided by the total of true positives and false positives.
- **Recall:** Indicates how well the model identifies all positive samples, computed as true positives divided by the total of true positives and false negatives.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics, especially in imbalanced datasets.
  
## Model
In this project, I manually implemented the ResNet-18 model (Residual Network), which is one of the most widely used neural networks for image classification tasks. ResNet-18 uses residual layers, allowing the model to efficiently train deep networks by avoiding the vanishing gradient problem. These residual connections enable the network to learn increasingly complex features from images, which improves its accuracy.

The model consists of 18 layers, including 16 convolutional layers and two fully connected layers, and its architecture has been adapted to classify 10 types of fruits based on their visual features.

## Structure of the Resnet-18:
![ResNet-18](https://github.com/kizokubanczyk/Fruit-classification/blob/main/screenshots/Structure_of_the_Resnet-18.png)

## Predicting Individual Fruit Images
The project includes a feature that allows users to input an image of a fruit (from one of the 10 fruit types), and the trained model will output the predicted fruit category. This functionality enables real-time classification of fruit images by processing the input and returning the most likely fruit label based on the visual characteristics.

To use this feature, simply provide an image, and the model will classify it as one of the following fruit types: apple, banana, orange, etc.

<img src="https://github.com/kizokubanczyk/Fruit-classification/blob/main/data/external/external.png" alt="external" width="500"/>
<img src="https://github.com/kizokubanczyk/Fruit-classification/blob/main/scores/external_images/image_1.jpeg" alt="external" width="500"/>

 ## To improve the model's performance
- I increased the training data size four times and applied augmentation to each image.
- I added an EarlyStopping class that saves the best model based on validation loss.
- I added a scheduler that monitored the validation loss and reduced the learning rate if it did not improve for 5 epochs.

## Scores
After these improvements, I was able to achieve the following results:
- Accuracy: 97.59%
- F1 Score: 97.60%
- Precision: 97.69%
- Recall: 97.59%
- Confusion Matrix:
```python
Confusion Matrix:
[[41,  1,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0, 34,  1,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0, 37,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  1,  0, 29,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0, 29,  0,  0,  0,  1,  0],
 [ 0,  0,  0,  0,  0, 34,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0,  0, 23,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  1, 28,  0,  1],
 [ 0,  0,  0,  0,  0,  0,  0,  0, 43,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 26]]
