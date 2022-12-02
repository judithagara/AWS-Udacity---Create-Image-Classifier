# Imports here
import torch
from torch import nn, optim
from torchvision import models

# my functions import
from train_functions import get_train_args, train_and_validate, validate_test_set, save_checkpoint
from image_processing import process_images

# parse command line arguments
train_args = get_train_args()

# transform and load images
train_data, trainloader, validloader, testloader = process_images(train_args.data_dir)

# TODO: Build and train your network
choose_model = {'resnet': models.resnet18(pretrained=True),
                'alexnet': models.alexnet(pretrained=True),
                'vgg': models.vgg16(pretrained=True)}

model = choose_model[train_args.arch]

# get input size
if train_args.arch == "vgg":
    input_size = model.classifier[0].in_features
elif train_args.arch == "alexnet":
    input_size = model.classifier[1].in_features
elif train_args.arch == "resnet":
    input_size = model.fc.in_features

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(nn.Linear(input_size, train_args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.8),
                           nn.Linear(train_args.hidden_units, 102),
                           nn.LogSoftmax(dim=1))
    
# Build and replace classifier, define the criterion and optimizer
if train_args.arch == "vgg" or train_args.arch == "alexnet":
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=train_args.lr)
elif train_args.arch == "resnet":
    model.fc = classifier
    optimizer = optim.Adam(model.fc.parameters(), lr=train_args.lr)
    


# Implement training and validation
train_and_validate(model, optimizer, train_args.epochs, trainloader, validloader, train_args.device)

# Validate test set
validate_test_set(model, testloader, train_args.device)

# Save checkpoint
model.class_to_idx = train_data.class_to_idx
save_checkpoint(train_args, model, classifier, optimizer)