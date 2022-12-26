import sys
sys.path.append('/home/hoo7311/anaconda3/envs/pytorch/lib/python3.8/site-packages')
import argparse
import os
import cv2
import xml.etree.ElementTree as ET
from glob import glob
from tqdm.auto import tqdm


classes = {
    'bicycle': [0, (255, 0, 0)],
    'bus': [1, (128, 0, 0)],
    'car': [2, (255, 255, 0)],
    'carrier': [3, (128, 128, 0)],
    'cat': [4, (0, 255, 0)],
    'dog': [5, (0, 128, 0)],
    'motorcycle': [6, (0, 255, 255)],
    'movable_signage': [7, (0, 128, 128)],
    'person': [8, (0, 0, 255)],
    'scooter': [9, (0, 0, 128)],
    'stroller': [10, (255, 0, 255)],
    'truck': [11, (128, 0, 128)],
    'wheelchair': [12, (255, 127, 80)],
    
    'barricade': [13, (184, 134, 11)],
    'bench': [14, (127, 255, 0)],
    'bollard': [15, (0, 191, 255)],
    'chair': [16, (255, 192, 203)],
    'fire_hydrant': [17, (165, 42, 42)],
    'kiosk': [18, (210, 105, 30)],
    'parking_meter': [19, (240, 230, 140)],
    'pole': [20, (245, 245, 220)],
    'potted_plant': [21, (0, 100, 0)],
    'power_controller': [22, (64, 224, 208)],
    'stop': [23, (70, 130, 180)],
    'table': [24, (106, 90, 205)],
    'traffic_light': [25, (75, 0, 130)],
    'traffic_light_controller': [26, (139, 0, 139)],
    'traffic_sign': [27, (255, 20, 147)],
    'tree_trunk': [28, (138, 43, 226)],
}

def voc2yolo_bbox(bbox, h, w):
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def convert_voc2yolo(path):
    folders = sorted(glob(path+'Bbox_*/Bbox_*'))
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
            label_folder_name = folder.replace(folder_name, folder_name.replace('Bbox_', 'Label_'))
            os.makedirs(label_folder_name, exist_ok=True)
            # create annotation file
            if file_name.split('.')[-1] == 'jpg':
                annotation_file = label_folder_name+'/'+file_name.replace('.jpg', '.txt')
            elif file_name.split('.')[-1] == 'png':
                annotation_file = label_folder_name+'/'+file_name.replace('.png', '.txt')
            else:
                raise ValueError(f'format {file_name} dose not support')
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
        print('complete...!', folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert format', add_help=False)
    parser.add_argument('--path', type=str, help='a path for converting formats')
    args = parser.parse_args()
    convert_voc2yolo(args.path)
