# Imports
import argparse
import torch
from torch import nn, optim
from torchvision import models
import os


def get_train_args():
    """
    Retrieves and parses the command line arguments provided by the user for
    training a network. If the user fails to provide some or all of the
    arguments, then the default values will be used for the missing arguments.
    This function returns these arguments as an ArgumentParser object.

    Command Line Arguments:
      1. Dataset Directory as data_dir with default as 'ImageClassifier/flowers'
      2. Checkpoint Folder as --save_dir with default value 'saved_models/'
      3. CNN Model Architecture as --arch with default value 'resnet' (options: resnet, alexnet or vgg)
      4. Learning Rate as --lr with default value 0.001
      5. Number of Hidden Neurons as --hidden_units with default value 307
      6. Number of Training Cycles as --epochs with default value 15
      7. Device to use for training as --device with default value 'gpu'


    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=str, default='ImageClassifier/flowers',
                        help="directory of training dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models/",
                        help="directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="resnet",
                        help="the CNN model architecture to use(resnet, alexnet or vgg)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="model learning rate")
    parser.add_argument("--hidden_units", type=int, default=307,
                        help="number of hidden units/neurons")
    parser.add_argument("--epochs", type=int, default=15,
                        help="number of training cycles")
    parser.add_argument("--device", type=str, default="gpu",
                        help="device to use for training(gpu or cpu)")

    return parser.parse_args()


def train_and_validate(model, optimizer, epochs, trainloader, validloader, device):
    """
    Implements training and validation loop
    """
    criterion = nn.NLLLoss()
    epochs = epochs
    steps = 0
    train_losses = []
    valid_losses = []
    print_every = 5
    # set device to use for training - use GPU if it's available
    device = torch.device("cuda" if device == "gpu" else "cpu")
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                # validation loss and accuracy
                model.eval()
                valid_loss = 0
                accuracy = 0

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model(images)
                        loss = criterion(log_ps, labels)
                        valid_loss += loss.item()

                        actual_probs = torch.exp(log_ps)
                        top_p, top_class = actual_probs.topk(1, dim=1)
                        is_pred_equal = top_class == labels.view(
                            *top_class.shape)
                        accuracy += torch.mean(
                            is_pred_equal.type(torch.FloatTensor)).item()

                # record training and validation losses
                train_losses.append(running_loss/len(trainloader))
                valid_losses.append(valid_loss/len(validloader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train Loss: {running_loss/len(trainloader):.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation Accuracy: {accuracy/len(validloader)*100:.3f}%")
                model.train()


def validate_test_set(model, testloader, device):
    """
    Performs validation on the test set
    """
    device = torch.device("cuda" if device == "gpu" else "cpu")
    model.to(device)
    criterion = nn.NLLLoss()
    model.eval()
    test_loss = 0
    accuracy = 0

    # Turn off gradients
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            test_loss += loss.item()

            actual_probs = torch.exp(log_ps)
            top_p, top_class = actual_probs.topk(1, dim=1)
            is_pred_equal = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(is_pred_equal.type(torch.FloatTensor)).item()

    print(f"Test Loss: {test_loss/len(testloader):.3f}.. "
          f"Test Accuracy: {accuracy/len(testloader)*100:.3f}%")


def save_checkpoint(train_args, model, classifier, optimizer):
    """
    saves trained model
    """
    choose_model = {'resnet': models.resnet18(pretrained=True),
                    'alexnet': models.alexnet(pretrained=True),
                    'vgg': models.vgg16(pretrained=True)}

    checkpoint = {'pretrained_model': choose_model[train_args.arch],
                  'model_fc': classifier,
                  'state_dict': model.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'no_of_epochs': train_args.epochs}

    torch.save(checkpoint, train_args.save_dir + 'checkpoint.pth')
