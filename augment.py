"""
This script provides image augmentation for labeled images based on
[CLoDSA](https://github.com/joheras/CLoDSA) library.
It supports 'coco', 'yolo', and  'pascal' bbox formats. augment.yml
is used to configure input and output formats of the augmentor and
the path to output directory. techniques.json specifies the augmentation
techniques that will be applied on the images and bounding boxes if applicable.
The processed files are stored in `$OUTPUT_MODE +'_augmented'`.

"""

from matplotlib import pyplot as plt
import cv2
import os
import shutil
import glob

from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique

import argparse
import configparser
import json
# import subprocess

# input arguments
def msg(name=None):
    return """
        python augment.py -d <input directory>
        """


parser = argparse.ArgumentParser('example:', usage=msg())
parser.add_argument('-d', '--dir', help='yolo format input directory', dest="inputpath")

args = vars(parser.parse_args())

# clodsa needs images and bbox .txt files to be in the same directory
cwd = os.getcwd()
inputpath = cwd + '/' + args["inputpath"]
temp = cwd + '/tmp'

if os.path.exists(temp):
  shutil.rmtree(temp)

os.makedirs(temp)

for path, subdirs, files in os.walk(inputpath):
    for name in files:
      if name.lower().endswith(tuple(['.jpg', '.jpeg', '.png', 'txt', '.xml'])):
        filename = os.path.join(path, name)
        shutil.copy2(filename, temp)


# configure the augmentor
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('./augment.yml')

PROBLEM = config.get('Augmentor','PROBLEM')
ANNOTATION_MODE = config.get('Augmentor','ANNOTATION_MODE')
INPUT_PATH = temp
GENERATION_MODE = config.get('Augmentor','GENERATION_MODE')
OUTPUT_MODE = config.get('Augmentor','OUTPUT_MODE')
OUTPUT_PATH = config.get('Augmentor','OUTPUT_PATH')


print("\nNumber of input images")
cmd = f"ls -1 {inputpath}/images/*.jpg | wc -l"
os.system(cmd)


# Creating the augmentor object
augmentor = createAugmentor(
    PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,
    INPUT_PATH,{"outputPath":OUTPUT_PATH})

# Instantiate the transformer
transformer = transformerGenerator(PROBLEM)

# add techniques to the transformer
with open("./techniques.json") as f:
    tech_json = json.load(f)


tech_dic = {
    "average_blurring": createTechnique("average_blurring",
                                        {"kernel": tech_json.get("average_blurring", {}).get("kernel", 5)}),
    "change_to_hsv": createTechnique("change_to_hsv",{}),
    "change_to_lab": createTechnique("change_to_lab",{}),
    "crop": createTechnique("crop",{"percentage": tech_json.get("crop", {}).get("percentage", 0.8),
                                    "startFrom": tech_json.get("crop", {}).get("startFrom", "TOPLEFT")}),
    "dropout": createTechnique("dropout",{"percentage": tech_json.get("dropout", {}).get("percentage", 0.05)}),
    "elastic": createTechnique("elastic",{"alpha": tech_json.get("elastic", {}).get("alpha",5),
                                          "sigma": tech_json.get("elastic", {}).get("sigma", 0.05)}),
    "equalize_histogram": createTechnique("equalize_histogram",{}),
    "vflip": createTechnique("flip",{"flip":0}),
    "hflip": createTechnique("flip", {"flip": 1}),
    "gamma": createTechnique("gamma",{"gamma": tech_json.get("gamma", {}).get("gamma", 1.5)}),
    "gaussian_noise": createTechnique("gaussian_noise", {"mean": tech_json.get("gaussian_noise", {}).get("mean", 0),
                                                         "sigma": tech_json.get("gaussian_noise", {}).get("sigma",10)}),
    "invert": createTechnique("invert",{}),
    "none": createTechnique("none",{}),
    "raise_hue": createTechnique("raise_hue", {"power": tech_json.get("raise_hue", {}).get("power", 0.9)}),
    "resize": createTechnique("resize", {"percentage": tech_json.get("resize", {}).get("percentage", 0.9),
                                         "method": tech_json.get("resize", {}).get("method", "INTER_NEAREST")}),
    "rotate": createTechnique("rotate", {"angle": tech_json.get("rotate", {}).get("angle", 90)}),
    "shearing": createTechnique("shearing", {"a": tech_json.get("shearing", {}).get("a", 0.5)}),
    "translation": createTechnique("translation", {"x": tech_json.get("translation", {}).get("x", 10),
                                                   "y": tech_json.get("translation", {}).get("y", 10)}),
}

for k in tech_json:
    augmentor.addTransformer(transformer(tech_dic[k]))

# Applying the augmentation process
augmentor.applyAugmentation()

print(f"{len(augmentor.transformers)} transformers complete.")
print("\n# of generated images:")
cmd = f"ls -1 {OUTPUT_PATH}/*.jpg | wc -l"
os.system(cmd)

#restructure augmented images and labels
result = cwd + '/' + OUTPUT_MODE +'_augmented'

if os.path.exists(result):
  shutil.rmtree(result)

os.makedirs(result)
os.makedirs(result + "/images")
os.makedirs(result + "/labels")

for filename in glob.glob(OUTPUT_PATH + '/*'):
    basename = os.path.basename(filename)
    if basename.lower().endswith(('.jpg', '.jpeg', '.png')):
        shutil.copyfile(filename, os.path.join(result + "/images", basename))
    elif filename.lower().endswith(('.txt','xml')):
        shutil.copyfile(filename, os.path.join(result + "/labels", basename))


shutil.rmtree(temp)
shutil.rmtree(OUTPUT_PATH)

