# TransDeepLab For Needle Biopsy Image Segmentation

(THIS IS A FORK OF THE ORIGINAL [TRANSDEEPLAB REPOSITORY](https://github.com/rezazad68/transdeeplab), ALL CREDITS GO TO THEM FOR THEIR ORIGINAL IMPLEMENTATION)

## Few notes about usage:

### `runImageThroughModel.py`

This python script produces an image segmentation mask with the trained ML model based on an image that you feed into it. This is how you use the script:
```bash
python3 script.py --image_path "mask_generation/fun_images/217613113.png"
```

### `mask_maker_script.py`

This python script produces masks based on a given annotation file. This is how you use it:
```bash 
python script_name.py --image_dir ./mask_generation/fun_images --annotation_file ./mask_generation/annotations1.csv --mask_dir ./mask_generation/masks
``` 
The only required argument is the annotation file. 