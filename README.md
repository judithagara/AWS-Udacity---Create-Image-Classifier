## AWS/Udacity Scholarship
#### Project 2: Create Your Own Image Classifier

This Project consists of two parts:

### 1. Developing an Image Classifier with Deep Learning
In this first part of the project, I worked through a Jupyter notebook to implement an image classifier with PyTorch. I built and trained a deep neural network on the [flower data set](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

### 2 - Building the Command Line Application
In this part of the project, I converted my trained network into a command line application that others can use. My application is a pair of Python scripts that run from the command line. I used the argparse module in the standard library to get the command line input into the scripts and my saved checkpoint in the first part as default arguments.

**Specifications:**
This part has two main files train.py and predict.py and 3 supporting scripts all contained in the repository. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class (flower specie) for an input image.
