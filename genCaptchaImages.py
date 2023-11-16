import os 
import cv2 
import random
import argparse 
import numpy as np
from tqdm import tqdm
from captcha.image import ImageCaptcha
from settings import IMG_H, IMG_W, MAX_LEN, LETTERS 

image_directory = "captchas"
os.makedirs(image_directory, exist_ok=True)

def randomize_content(lower_len, upper_len):
    content = ''.join([random.choice(LETTERS[4:]) for _ in range(random.randint(lower_len, upper_len))])
    return content

def main(args):
    imageCaptcha = ImageCaptcha(height = IMG_H, width = IMG_W)

    capt_count = {}
    for i in tqdm(range(args.num)):
        contents = [randomize_content(lower_len=2, upper_len=5) for _ in range(args.lines)]
        joined_content = ''.join(contents)
        # keep track of the captcha count for each content
        if joined_content not in capt_count:
            capt_count[joined_content] = 0 
        capt_count[joined_content] += 1

        filename = f"{joined_content}-{capt_count[joined_content]}"
        capt_imgs = [imageCaptcha.generate_image(c) for c in contents]

        # concatenate images vertically using numpy
        capt_img = capt_imgs[0]
        for i in range(1, len(capt_imgs)):
            capt_img = np.concatenate((capt_img, capt_imgs[i]), axis=0)

        # save data 
        cv2.imwrite(os.path.join(image_directory, f"{filename}.png"), capt_img)
        # capt_img.save(os.path.join(image_directory, f"{filename}.png"))
        with open(os.path.join(image_directory, f"{filename}.txt"), "w") as f:
            f.write(joined_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-num',
                        '--num',
                        type=int,
                        required=True,
                        help="Number of images to generate")

    parser.add_argument('-lines',
                        '--lines',
                        type=int,
                        default=1,
                        required=False,
                        help="Number of text lines")
    
    args = parser.parse_args()
    main(args)