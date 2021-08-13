"""
This script converts Pascal VOC labeled images to yolo format

dirs: list of pascal voc directories
img_dir : image directory name under pascal voc directory
xml_dir : xml directory under pascal voc directory

classes: list of classes of interest to be converted to yolo format
ext: The existing image extension

output directory: /yolo_format/ is created under each input directory
with ./labels and ./images under it.

png files are converted to jpg as well.
"""

import glob
import os
# import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
import argparse
from PIL import Image

def msg(name=None):
    return """
        python voc2yolo.py -d pascal_voc -i JPEGImages
        -l Annotations -c Bird Drone Quadcopter -e PNG
        """


parser = argparse.ArgumentParser('example:', usage=msg())
parser.add_argument('-d', '--dirs', nargs='*', help='space separated list of pascal voc directories', dest="dirs")
parser.add_argument('-i', '--imgd', help='image directory', dest="img_dir")
parser.add_argument('-l', '--xml', help='xml directory', dest="xml_dir")
parser.add_argument('-c', '--classes', nargs='*', help='list of desired classes space separated', dest="classes")
parser.add_argument('-e', '--ext', help='image extension', dest="ext")

args = parser.parse_args()


# dirs = ['train', 'val']
dirs = ['pascal_voc']
img_dir = 'JPEGImages'
xml_dir = 'Annotations'

# classes to be considered for conversion
classes = ['Bird', 'Drone', 'Quadcopter']
ext = 'PNG'

def getImagesInDir(img_path, extension):
    image_list = []
    for filename in glob.glob(img_path + '/*.' + extension):
        image_list.append(filename)

        if extension.lower() == "png":
            basename = os.path.basename(filename)
            basename_no_ext = os.path.splitext(basename)[0]

            im1 = Image.open(img_path + '/' + basename)
            im1.save(output_path + 'images/' + basename_no_ext + '.jpg')

    return image_list

def convert(size, box):
    """
    size = (image width, image height)
    box = (xmin, xmax, ymin, ymax)
    return:
    (x_center, y_center, width, height)
    0 <= x_center, y_center <= 1
    """
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_path, output_path, img_name):
    basename = os.path.basename(img_name)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(xml_path + '/' + basename_no_ext + '.xml')
    out_file = open(output_path + 'labels/' + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    # image size field in xml file
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # object refers to each labeled object in xml file
    # filter out classes of difficult or not listed in classes variable
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(round(a, 4)) for a in bb]) + '\n')
    # out_file.close()
    # in_file.close()

cwd = getcwd()

for dir_path in dirs:
    full_dir_path = cwd + '/' + dir_path
    img_dir_path = full_dir_path + '/' + img_dir
    xml_dir_path = full_dir_path + '/' + xml_dir

    output_path = full_dir_path +'/yolo_format/'
    output_label_path = output_path + 'labels/'
    output_image_path = output_path + 'images/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(output_label_path):
        os.makedirs(output_label_path)

    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    image_names = getImagesInDir(img_dir_path, ext)
    list_file = open(full_dir_path + '.txt', 'w')

    for image_name in image_names:
        list_file.write(image_name + '\n')
        convert_annotation(xml_dir_path, output_path, image_name)
    list_file.close()

    print("Finished processing: " + dir_path)
