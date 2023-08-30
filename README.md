
# Utilizing Convolutional Neural Networks for Horses vs. Humans Classification

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/horses_and_humas.png?raw=true)

## Introduction

This is a computer vision model that uses CNN (Convolutional Neural Networks) and DNN (Deep Neural Networks) to recognize and classify horses and humans from the computer-generated images.


## About Dataset

The model uses the horse or human dataset from the Kaggle data science platform. It is a dataset of 300×300 images, created by Laurence Moroney, that is licensed CC-By-2.0. This dataset is made for anybody to use in learning or testing computer vision algorithms.

The set contains 500 rendered images of various species of horses in various poses in various locations. It also contains 527 rendered images of humans in various poses and locations. Emphasis has been taken to ensure the diversity of humans, and to that end, there are both men and women as well as Asian, Black, South Asian, and Caucasians present in the training set. The validation set adds 6 different figures of different genders, races, and poses to ensure the breadth of data.

## Basic Model

Keras API from TensorFlow was used to build the CNN. To begin with, the model was built with 5 convolutional layers and 5 max-pooling layers followed by 2 dense layers. The "Relu" activation function was used in every convolution layer and dense layer except for the last layer which uses the sigmoid activation function. Since this is a binary classification sigmoid function is used to output the result as either 0 or 1.


> Preview of the model

    model = tf.keras.models.Sequential([
	    keras. layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (300,300,3)),
	    keras.layers.MaxPooling2D(2,2),
	    keras.layers.Conv2D(32, (3,3), activation = 'relu'),
	    keras.layers.MaxPooling2D(2,2),
	    keras.layers.Conv2D(64, (3,3), activation = 'relu'),
	    keras.layers.MaxPooling2D(2,2),
	    keras.layers.Conv2D(64, (3,3), activation = 'relu'),
	    keras.layers.MaxPooling2D(2,2),
	    keras.layers.Conv2D(64, (3,3), activation = 'relu'),
	    keras.layers.MaxPooling2D(2,2),
	    keras.layers.Flatten(),
	    keras.layers.Dense(512, activation = 'relu'),
	    keras.layers.Dense(1, activation = 'sigmoid')
	])

> model summary

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/summary1.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/summary2.png?raw=true)


The model's summary specifies the layer's name, output shape, and associated parameters. Since the model is not trained before there are no non-trainable parameters.


## Visualization of Convolutional layers

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l1.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l2.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l3.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l4.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l5.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l6.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l7.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l8.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l9.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/l10.png?raw=true)

Each convolution layer contains a specified number of filters and they are used to extract the features from the images. The extracted features are flattened and passed to the DNN layer. After the last convolutional layer, the number of parameters is significantly reduced with extracted features.

## Evaluation

The dataset is divided into training and validation folders and each folder contains folders of classification labels which in turn contain images for the associated labels.

Image data generators are used to preprocess the images before training the data. Here we only scale the data for every pixel of the image. Since every pixel values from 0 to 255 dividing it by 255 converts range from 0 to 1. Image data generators are also used to restrict the input of data, we can constrain the batch size in image data generator to avoid memory overloading. Two image generators are used one for the training set and the other for the validation set.

The model is trained under the given dataset for 100 epochs. At each epoch, the model calculates the accuracy and loss for the training set as well as for the validation set.

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/accuracy_without_augmentation.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/loss_without_augmentation.png?raw=true)

It specifies that the model is overfitted since there is no increase in validation accuracy for longer iterations.

The accuracy of the model is approximated to be 84% accurate.

Some of the images and their predictions.

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/human1.png?raw=true)
***Predicted as human***

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/human2.png?raw=true)
***Predicted as human***

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/horse1.png?raw=true)
***Predicted as human***

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/horse2.png?raw=true)
***Predicted as horse***

## Image Augmentation

Image augmentation is a technique used in computer vision and machine learning to artificially increase the size and diversity of a training dataset by applying various transformations and modifications to existing images. The goal of image augmentation is to improve the generalization and robustness of machine learning models by exposing them to a wider range of variations that might occur in real-world scenarios.

Image augmentation is particularly useful when working with limited datasets. By applying these transformations to the existing images, the dataset effectively grows in size, reducing the risk of overfitting and allowing the model to generalize better to new, unseen data.

> Parameters used for image augmentation

    train_datagen_aug = ImageDataGenerator(
	    rotation_range=40,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    shear_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=True,
	    vertical_flip=False,
	    fill_mode = 'nearest',
	    rescale=1.0 / 255
	)

The model is trained on augmented data to address overfitting and increase accuracy. The layers of the model are not modified. The training of the model is slow due to an increase in images after augmentation for each epoch.

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/accuracy_with_augmentation.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/loss_with_augmentation.png?raw=true)

From the results, it is very clear that the image augmentation didn't help the model to address the issue of overfitting and it resulted in an increase of noise in the accuracy and loss validation sets.

It is because the image augmentation is not suitable for every dataset and it may even result in worse accuracy. To check the performance of augmentation on the other dataset, the cats vs. dogs dataset is used. The same model was used to train the model on the cats vs. dogs dataset with augmentation.

> Accuracy of cats vs. dogs dataset with augmentation

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/catsvsdogs_accuracy.png?raw=true)
> Loss of cats vs. dogs dataset with augmentation

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/catsvsdogs_loss.png?raw=true)

From the above graph, it is clear that the accuracy of the validation set is steadily increasing and it has addressed the issue of overfitting. Augmentation seemed to work well for the cats vs. dogs rather than horses vs. humans dataset.
It is to be well noted that the parameters for the augmentation have to be experimented with to obtain better results.

## Transfer Learning using Inception_v3 Model

Inception_v3 is a model which is trained on the ImageNet dataset which comprises 1.3 million images. This model is best known for its performance in extracting the features from the images and classifying them. The adapted features from this model can be used on a task with smaller datasets.

The features of the model can be extracted using transfer learning. Transfer learning is a machine learning technique where a pre-trained model developed for one task is reused as a starting point for a model on a second task. Instead of training a model from scratch on the second task, transfer learning leverages the knowledge gained during the first task to improve the performance on the second task, especially when the second task has limited labeled data available.

The convolutional layers from the Inception_v3 are taken with already trained parameters and additional dense layers are added to make a complete model.

    model_hub = tf.keras.Sequential([
	    tf.keras.layers.InputLayer(input_shape = (300, 300, 3)),
	    hub.KerasLayer('/kaggle/input/inception-v3/tensorflow2/classification/2', trainable = False),
	    tf.keras.layers.Flatten(),
	    tf.keras.layers.Dropout(0.2),
	    tf.keras.layers.Dense(1, activation = 'sigmoid')
	])

> Summary of the model

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/inceptionv3_summary.png?raw=true)

It can be observed that there are non-trainable parameters because the inception_v3 model has already been trained over a large dataset and those trained parameters from convolution layers are taken as reference and the custom model is made. The trainable parameters are those from dense layers which are newly added.

## Analysis of Inception_v3 model

An additional dropout layer is added to avoid redundancy weights and conflicts between neurons in dense layers. This helps in reducing the overfitting of data.

The model is trained with augmented data and it is trained for 25 epochs.

![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/inception_accuracy.png?raw=true)
![enter image description here](https://github.com/Satzil/horses_vs_humans/blob/main/images/inception_loss.png?raw=true)
 
The accuracy of the model turned out to be better than previous models and it has almost 100% accuracy over the dataset. The time taken to train the custom inception model was comparatively more than the previous models so far because it consists of more convolution layers.

## Conclusion

Classification models have to be built according to the dataset given. Parameters for the augmentation have to be experimented with to obtain better results. A dropout layer has to be added to avoid redundancy of weights between neurons. Make use of already trained models like inception_v3 for better performance.

### Resources
[Horses vs. Humans Notebook](https://www.kaggle.com/code/satzil/tensorflow-2)
[Cats vs. Dogs Notebook](https://www.kaggle.com/code/satzil/fork-of-tensorflow-2)


 
