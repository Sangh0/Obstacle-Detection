import sys
sys.path.append('/home/hoo7311/anaconda3/envs/obstacle/lib/python3.8/site-packages')
import os
import cv2
import json
import xml.etree.ElementTree as ET
import argparse
from datetime import datetime
from glob import glob
from tqdm.auto import tqdm


classes = {
    'bicycle': 0,
    'bus': 1,
    'car': 2,
    'carrier': 3,
    'cat': 4,
    'dog': 5,
    'motorcycle': 6,
    'movable_signage': 7,
    'person': 8,
    'scooter': 9,
    'stroller': 10,
    'truck': 11,
    'wheelchair': 12, 
    
    'barricade': 13,
    'bench': 14,
    'bollard': 15,
    'chair': 16,
    'fire_hydrant': 17,
    'kiosk': 18,
    'parking_meter': 19,
    'pole': 20,
    'potted_plant': 21,
    'power_controller': 22,
    'stop': 23,
    'table': 24,
    'traffic_light': 25,
    'traffic_light_controller': 26,
    'traffic_sign': 27,
    'tree_trunk': 28, 
}

class_names = [class_name for class_name in classes.keys()]

def base_json(class_names):
    
    json_file = {
        "info": {
            "year": "2022",
            "version": "01",
            "description": "data from ai hub",
            "contributor": "ai hub",
            "url": "https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=189",
            "data_created": "2022-11-14"
        },
        
        "licenses": [
            {
                "id": 1,
                "url": "https://www.aihub.or.kr",
                "name": "ai hub"
            }
        ],
        
        "categories": [
            {
                "id": idx,
                "name": class_names[idx],
                "supercategory": "none"
            } for idx in range(len(class_names))
        ],
        
        "images": [],
        
        "annotations": []
    }
    
    return json_file


def images_json(base_json, img_dir_list):
    base_json["images"] = [
        {
            "id": idx,
            "license": 1,
            "file_name": get_img_info(img_dir)['file_name'],
            "height": get_img_info(img_dir)['height'],
            "width": get_img_info(img_dir)['width'],
            "date_captured": get_img_info(img_dir)['date']
        } for idx, img_dir in enumerate(img_dir_list)
    ]
    
    return base_json


def annotations_json(base_json, idx, image_idx, category_idx, bbox):
    base_json["annotations"] = [
        {
            "id": idx,
            "image_id": image_idx,
            "category_id": category_idx,
            "bbox": [
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3]
            ],
            "area": bbox[2]*bbox[3],
            "segmentation": [],
            "iscrowd": 0
        }
    ]
    
    return base_json


def get_img_info(img_dir):
    img = cv2.imread(img_dir)
    height, width, _ = img.shape
    now = datetime.now()
    date = str(now.year)+'-'+str(now.month)+'-'+str(now.day)
    return {
        'file_name': img_dir,
        'height': height,
        'width': width,
        'date': date,
    }


def convert_bbox_voc2yolo(label):
    # voc format
    x1, y1 = int(float(label.get('xtl')))-1, int(float(label.get('ytl')))-1
    if x1 < 0 or y1 < 0:
        x1, y1 = 0, 0
    x2, y2 = int(float(label.get('xbr'))), int(float(label.get('ybr')))
    assert x1 < x2 and y1 < y2, \
        f'x1 {x1} is bigger than x2 {x2} or y1 {y1} is bigger than y2 {y2}'
    # convert voc to coco format
    x, y = x1, y1
    w, h = x2 - x1, y2 - y1
    return x, y, w, h


def load_xml(path, class_names):
    xml_list = sorted(glob(path+'**/**/*.xml'))

    for xml in tqdm(xml_list):
        coco_format_base = base_json(class_names)

        common_path = '/'.join(xml.split('/')[:-1])
        file_name = common_path+'/'+xml.split('/')[-1].split('.')[0]+'.json'
        annotations = ET.parse(xml).getroot()
        image_list = annotations.findall('image')
        image_list_contain_alldir = [common_path+'/'+image.get('name') for image in image_list]

        # input image informations in coco file
        coco_format_json = images_json(coco_format_base, image_list_contain_alldir)
        
        anno_all_id_count = 0
        # an image
        for image_idx, image in enumerate(image_list):
            # annotations in an image
            for label_idx, label in enumerate(image):
                class_name = label.get('label')

                bbox_info = convert_bbox_voc2yolo(label)
                
                coco_format_json['annotations'].append(
                    {
                        "id": anno_all_id_count,
                        "image_id": image_idx,
                        "category_id": classes[class_name],
                        "bbox": [bbox_info[0], bbox_info[1], bbox_info[2], bbox_info[3]],
                        "area": bbox_info[2]*bbox_info[3],
                        "segmentation": [],
                        "iscrowd": 0
                    }
                )

                anno_all_id_count += 1

        print('save', file_name)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(coco_format_json, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Convert voc format to coco format')
    parser.add_argument('--path', type=str, required=True,
                        help='directory to convert voc to coco')
    args = parser.parse_args()

    load_xml(args.path, class_names)

if __name__ == "__main__":
    main()
