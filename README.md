# Labeled Images Augmentation

This repo provides image augmentation for labeled images 
in object detection using
[CLoDSA](https://github.com/joheras/CLoDSA) library.
It supports 'coco', 'yolo', and  'pascal' bbox formats. 
`augment.yaml` is used to configure the augmentor including 
the input and output formats, and generation mode. 
`techniques.json` specifies the augmentation techniques that 
will be applied on the images and bounding boxes if applicable.
The processed files are stored in `$OUTPUT_MODE +'_augmented'`.

### Installation
Dependencies are installed in a conda environment. After cloning
the repo activate the conda environment using:
```
conda env create -f environment.yml
```

After configuring augment.yml and techniques.json, you can generate
augmented images and bounding boxes using:
```
python augment.py -d <input directory>
``` 
Augmented data is stored in `$OUTPUT_MODE +'_augmented'`.

