import sys
sys.path.append('/home/hoo7311/anaconda3/envs/copy_torch/lib/python3.8/site-packages')
import os
import copy
import argparse
import pandas as pd
from glob import glob
from tqdm.auto import tqdm


classes_names = {
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

inv_classes_names = {v:k for k, v in classes_names.items()}

classes = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
    '6': 0,
    '7': 0,
    '8': 0,
    '9': 0,
    '10': 0,
    '11': 0,
    '12': 0,
    '13': 0,
    '14': 0,
    '15': 0,
    '16': 0,
    '17': 0,
    '18': 0,
    '19': 0,
    '20': 0,
    '21': 0,
    '22': 0,
    '23': 0,
    '24': 0,
    '25': 0,
    '26': 0,
    '27': 0,
    '28': 0,    
}


def count_classes(path, category='detail'):
    folders = sorted(glob(path+'Bbox_*'))
    if category == 'detail':
        folders = [sorted(glob(folder+'/Label_*')) for folder in folders]
        folders = sum(folders, [])
    elif category == 'large':
        folders = folders
    else:
        raise ValueError(f'{category} does not exists')

    total_class_count_dict = {}
    for folder in tqdm(folders):
        class_count_copy = copy.deepcopy(classes)
        if category == 'large':
            files = glob(folder+'/Label_*/*.txt')
        else:
            files = glob(folder+'/*.txt')
        for file in files:
            with open(file, 'r') as f:
                for anno in f.readlines():
                    for i, sub_anno in enumerate(anno.rstrip().split(' ')):
                        if i == 0:
                            class_count_copy[sub_anno] += 1
        total_class_count_dict[folder] = list(class_count_copy.values())
        print('complete...!', folder)
    return total_class_count_dict


def create_csv(class_count_dict, save_file_name):
    df = pd.DataFrame(class_count_dict)
    df.rename(index=inv_classes_names, inplace=True)
    df.to_csv(save_file_name)


def main():
    parser = argparse.ArgumentParser(description='save class counts to split dataset')
    parser.add_argument('--folder_category', type=str, choices=['large', 'detail'],
                        help='select large of detail, if you are selected large, then run count of classes for large folders')
    parser.add_argument('--path', type=str, required=True,
                        help='your folder directory')
    parser.add_argument('--save_file_name', type=str, required=True,
                        help='a file name with csv format for saving in local')
    args = parser.parse_args()

    class_count_dict = count_classes(args.path, args.folder_category)
    create_csv(class_count_dict, args.save_file_name)
    print('saving directory: ', args.save_file_name)


if __name__ == '__main__':
    main()
