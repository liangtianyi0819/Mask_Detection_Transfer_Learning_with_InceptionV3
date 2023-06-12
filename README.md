# Mask Detection using Deep Learning

Project creator: [Tianyi Liang](https://www.linkedin.com/in/tianyi-liang-at-bu/)   

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Model Building and Training](#model-building-and-training)
- [Model Evaluation](#model-evaluation)
- [Built With](#built-with)
- [Results](#results)
- [Conclusion and Insights](#conclusion-and-insights)
- [References](#references)

## Introduction <a name="introduction"></a>
This project is centered on developing a Convolutional Neural Network (CNN) model to perform mask detection in images. The model leverages transfer learning using the InceptionV3 architecture and is trained and evaluated on a dataset of images featuring people with and without masks. This initiative is crucial in the backdrop of the ongoing COVID-19 pandemic.

## Getting Started <a name="getting-started"></a>

### Prerequisites <a name="prerequisites"></a>
* Python 3.8
* Tensorflow 2.5+
* Numpy
* Matplotlib

## Data Preprocessing <a name="data-preprocessing"></a>
The data used for this project was sourced from the following:
[Face Mask Detection ~12K Images Dataset by Ashish Jangra](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) 

The dataset is preprocessed by resizing images, normalizing pixel values, and splitting the data into training, validation, and test sets. Dataset balance was also ensured to avoid biased results.

## Model Building and Training <a name="model-building-and-training"></a>
A model based on the InceptionV3 architecture is built, with a fully connected layer added to the end. The model is then trained using the training dataset, and the performance is monitored using the validation dataset.

## Model Evaluation <a name="model-evaluation"></a>
The model's performance is evaluated using metrics such as accuracy, Precision, Recall, and Specificity. Incorrect predictions are identified and analyzed to provide insights into the types of images the model has difficulties with.

## Built With <a name="built-with"></a>
* [Tensorflow](https://www.tensorflow.org/) - The framework used for building the model
* [Numpy](https://numpy.org/) - Used for numerical computation
* [Matplotlib](https://matplotlib.org/) - Used for data visualization

## Results <a name="results"></a>
The model demonstrated promising results, achieving high precision, recall, and specificity. It was noted that the model struggles with images containing anime icons or having very low resolution. 

## Conclusion and insights <a name="conclusion-and-insights"></a>
In this project, I developed a Convolutional Neural Network (CNN) model for mask detection using transfer learning based on the InceptionV3 architecture. The model was trained and evaluated on a dataset of images of people with and without masks.

The model demonstrated promising results, with a good balance between precision and recall. However, it's important to note that in the context of mask detection, missing an alarm (failing to detect a person not wearing a mask) is a more serious issue than a false alarm (incorrectly identifying a person as not wearing a mask when they are). Therefore, future improvements to the model should focus on minimizing the number of missed alarms.

Upon examining the instances where the model made incorrect predictions, I noticed that the model tends to struggle with images that contain anime icons or have very low resolution. This insight could guide future data collection and preprocessing efforts. For instance, we could aim to collect more high-resolution images and images that do not contain anime icons.

Balancing the dataset was a key challenge in this project. Ensuring that there is an equal number of images for both classes (mask and no mask) is crucial for avoiding biased results. The implementation of image augmentation techniques was also beneficial in increasing the diversity of the training data and preventing overfitting.

For future work, there are several potential directions for improving the model's performance. These include experimenting with different architectures, tuning the hyperparameters, and using more advanced techniques for handling imbalanced data.

In conclusion, this project has demonstrated the potential of deep learning for mask detection, a critical task in the context of the ongoing COVID-19 pandemic. It has also highlighted the importance of careful data preparation and model evaluation in achieving reliable and interpretable results. The insights gained from this project, particularly regarding the types of images that the model struggles with, will be valuable for further improving the model and ultimately enhancing its effectiveness in real-world applications.

## References <a name="references"></a>
Ashish Jangra (2021), "Face Mask Detection ~12K Images Dataset", Kaggle, available at: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset.
