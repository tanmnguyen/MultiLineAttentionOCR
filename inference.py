import cv2 
import torch 
import argparse 

from settings import settings
from utils.metrics import to_string
from utils.io import show_prediction
from utils.batch import img_processing
from utils.selections import get_model

def main(args):
    # load model 
    model, _, decode_fn = get_model(
        "attdec" if "attdec" in args.weight 
        else "crnn" if "crnn" in args.weight 
        else "GPT"
    )
    model.load_state_dict(torch.load(args.weight, map_location=settings.DEVICE))

    # get input image 
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_processing(img).float()
    img = img.unsqueeze(0)

    # decode prediction 
    pred = model(img)
    pred = decode_fn(pred)
    pred = to_string(pred[0])   

    show_prediction(img[0], pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-weight',
                        '--weight',
                        type=str, 
                        required=True,
                        help="Path to the model .pt weight")
    
    parser.add_argument('-image',
                        '--image',
                        type=str, 
                        required=True,
                        help="Path to an image file")
    
    parser.add_argument('-cfg',
                        '--cfg',
                        type=str,
                        required=False,
                        help="Path to the custom configuration .cfg  file")

    args = parser.parse_args()
    main(args)