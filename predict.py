# Imports here
import json

# my functions
from train_functions import get_train_args
from predict_functions import get_predict_args, load_checkpoint, process_image, predict

predict_args = get_predict_args()
loaded_model = load_checkpoint(predict_args.checkpoint)
print(loaded_model)

with open(predict_args.category_names, 'r') as f:
    cat_to_name = json.load(f)

probs, classes = predict(predict_args.image_path, loaded_model, predict_args.top_k, predict_args.device)
flower_names = [cat_to_name[e] for e in classes]
Result = list(zip(flower_names, probs))
print(Result)