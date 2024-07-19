import argparse
import torch
from PIL import Image
import numpy as np
import cv2
import os
from model.swin_deeplab import SwinDeepLab
import importlib

def preprocess_image(image_path, img_size):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    
    # Resize the image to the specified size (e.g., 224x224)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Normalize the image
    image = image / 255.0
    
    # Convert the image from HWC to CHW format
    image = image.transpose(2, 0, 1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict(model, image):
    with torch.no_grad():
        image = torch.tensor(image, dtype=torch.float32).cuda()
        outputs = model(image)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        return outputs.cpu().numpy()
    
def string_exploder(file_path):
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    return file_name_without_extension

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='swin_224_7_4level', help='config file name without .py extension')
    parser.add_argument('--ckpt_path', type=str, default='pretrained_ckpt/swin_224_7_4level.pth', help='path to checkpoint file')
    parser.add_argument('--image_path', type=str, required=True, help='path to input image')
    parser.add_argument('--img_size', type=int, default=224, help='input image size')
    parser.add_argument('--threshold', type=int, default=128, help='threshold for binary image')
    args = parser.parse_args()

    # Dynamically import the config
    config_module = importlib.import_module(f'model.configs.{args.config_file}')
    
    # Create the model
    model = SwinDeepLab(
        config_module.EncoderConfig,
        config_module.ASPPConfig,
        config_module.DecoderConfig
    ).cuda()

    # Load the checkpoint
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    # Preprocess and predict
    image = preprocess_image(args.image_path, img_size=args.img_size)
    mask = predict(model, image)

    # Get the number of classes from the configuration
    num_classes = config_module.DecoderConfig.num_classes

    # Save the mask as a grayscale image
    mask_image = (mask * 255 / (num_classes - 1)).astype(np.uint8)
    
    # Apply threshold to create black and white image
    _, bw_image = cv2.threshold(mask_image, args.threshold, 255, cv2.THRESH_BINARY)
    
    bw_image = Image.fromarray(bw_image)
    bw_image.save(f'model_generated_masks/{string_exploder(args.image_path)}_mask.png')
    print(f"Mask saved as model_generated_masks/{string_exploder(args.image_path)}_mask.png")
