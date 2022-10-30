import sys
import argparse
import os
import cv2
import xml.etree.ElementTree as ET
from glob import glob
from tqdm.auto import tqdm


classes = {
    'wheelchair': [0, (255, 0, 0)],
    'truck': [1, (128, 0, 0)],
    'tree_trunk': [2, (255, 255, 0)],
    'traffic_sign': [3, (128, 128, 0)],
    'traffic_light': [4, (0, 255, 0)],
    'traffic_light_controller': [5, (0, 128, 0)],
    'table': [6, (0, 255, 255)],
    'stroller': [7, (0, 128, 128)],
    'stop': [8, (0, 0, 255)],
    'scooter': [9, (0, 0, 128)],
    'potted_plant': [10, (255, 0, 255)],
    'power_controller': [11, (128, 0, 128)],
    'pole': [12, (255, 127, 80)],
    'person': [13, (184, 134, 11)],
    'parking_meter': [14, (127, 255, 0)],
    'movable_signage': [15, (0, 191, 255)],
    'motorcycle': [16, (255, 192, 203)],
    'kiosk': [17, (165, 42, 42)],
    'fire_hydrant': [18, (210, 105, 30)],
    'dog': [19, (240, 230, 140)],
    'chair': [20, (245, 245, 220)],
    'cat': [21, (0, 100, 0)],
    'carrier': [22, (64, 224, 208)],
    'car': [23, (70, 130, 180)],
    'bus': [24, (106, 90, 205)],
    'bollard': [25, (75, 0, 130)],
    'bicycle': [26, (139, 0, 139)],
    'bench': [27, (255, 20, 147)],
    'barricade': [28, (138, 43, 226)],
}

def voc2yolo_bbox(bbox, h, w):
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def convert_voc2yolo(path):
    folders = glob(path+'/images/**')
    for folder in tqdm(folders):
        xml_file = glob(folder+'/*.xml')
        if len(xml_file) != 1:
            print(folder, len(xml_file))
            raise ValueError('Must be exist one xml file')
        else:
            annotations = ET.parse(xml_file[0]).getroot()
            
        file_list = annotations.findall('image')
        
        for file in file_list:
            file_name = file.get('name')
            img_file_name = folder+'/'+file_name
            img = cv2.imread(img_file_name)
            img_height, img_width, _ = img.shape
            
            # create label folder
            folder_name = img_file_name.split('/')[-2]
            label_folder_name = folder.replace('images', 'labels')+'_label'
            os.makedirs(label_folder_name, exist_ok=True)
            
            # create annotation file
            annotation_file = label_folder_name+'/'+file_name.replace('.png', '.txt')
            f = open(annotation_file, 'w')
            
            label_num_count = 0
            for sub_file in file:
                # get label
                label = sub_file.get('label')
                label = classes[label]
                f.write(str(label[0])+' ')
                
                # get bbox coordinates
                x1, y1 = int(float(sub_file.get('xtl'))), int(float(sub_file.get('ytl')))
                x2, y2 = int(float(sub_file.get('xbr'))), int(float(sub_file.get('ybr')))
                bbox = [x1, y1, x2, y2]
                yolo_format = voc2yolo_bbox(bbox, img_height, img_width)
                for i, coor in enumerate(yolo_format):
                    if i == 3:
                        f.write(str(coor)+'\n')
                    else:
                        f.write(str(coor)+' ')
                label_num_count += 1
            f.close()
            print(f'label count:{label_num_count}', annotation_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert format', add_help=False)
    parser.add_argument('--path', type=str, help='a path for converting formats')
    args = parser.parse_args()
    convert_voc2yolo(args.path)
