import torch
import argparse
from torchvision import models, transforms
from PIL import Image


def load_checkpoint(filepath):
    """
    function to load saved checkpoint
    """
    checkpoint = torch.load(filepath)
    model = checkpoint['pretrained_model']
    if model.__class__.__name__ == "VGG" or model.__class__.__name__ == "AlexNet":
        model.classifier = checkpoint['model_fc']
    elif model.__class__.__name__ == "ResNet":
        model.fc = checkpoint['model_fc']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def get_predict_args():
    """
    Retrieves and parses the command line arguments provided by the user for
    predicting an image. If the user fails to provide some or all of the
    arguments, then the default values will be used for the missing arguments.
    This function returns these arguments as an ArgumentParser object.
    
    Command Line Arguments:
      1. Image Path for Prediction as image_path with default value "ImageClassifier/flowers/valid/102/image_08006.jpg"
      2. Checkpoint File as "checkpoint" with default value "saved_models/checkpoint.pth"
      3. Top K Classes to Predict as --top_k with default value 5
      4. File to map Category Names as --category_names with default value "ImageClassifier/cat_to_name.json"
      5. Device to use for training as --device with default value 'gpu'
    
    
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("image_path", type=str, default="ImageClassifier/flowers/valid/102/image_08006.jpg", help="path to image for prediction")
    parser.add_argument("checkpoint", type=str, default="saved_models/checkpoint.pth", help="path to checkpoint file")
    parser.add_argument("--top_k", type=float, default=5, help="return top_k classes")
    parser.add_argument("--category_names", type=str, default="ImageClassifier/cat_to_name.json", help="filepath for mapping of categories to real names")
    parser.add_argument("--device", type=str, default="gpu", help="device to use for training(gpu or cpu)")
    
    return parser.parse_args()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])])
    processed_image = process(image)
    
    return processed_image


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze(0).float()
    
    device = torch.device("cuda" if device == "gpu" else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        image = image.to(device)
        log_ps = model(image)
        ps = torch.exp(log_ps)
    
        top_p, top_idx = ps.topk(topk, dim=1)
        
        top_p = top_p.tolist()[0]
        top_idx = top_idx.tolist()[0]
        
        idx_to_class = {y:x for (x, y) in model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_idx]
    
    return top_p, top_class