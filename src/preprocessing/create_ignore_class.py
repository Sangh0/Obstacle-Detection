import sys
sys.path.append('/home/hoo7311/anaconda3/envs/pytorch/lib/python3.8/site-packages')
import os
import ast
import copy
import json
import argparse
from distutils.dir_util import copy_tree
from glob import glob
from tqdm.auto import tqdm


NUM_CLASSES = 29


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

inv_classes = {v:k for k, v in classes.items()}

integer_classes = {str(k): k for k in range(NUM_CLASSES)}


def arg_as_list(param):
    arg = ast.literal_eval(param)
    if type(arg) is not list:
        raise argparse.ArgumentTypeError('Argument \ "%s\" is not a list'%(arg))
    return arg


def get_args_parser():
    parser = argparse.ArgumentParser(description='Set ignore classes', add_help=False)
    parser.add_argument('--path', type=str, required=True,
                        help='the dataset dictionary') # /MY_DL/dataset/obstacle_detection
    parser.add_argument('--new_folder_name', type=str, required=True,
                        help='a new folder name') # 새로운 폴더 명
    parser.add_argument('--ignore_classes', type=arg_as_list, required=True,
                        help='a list of ignore classes') # 무시할 클래스 리스트로 입력
    return parser


def create_new_folder(folder_name: str):
    os.makedirs(folder_name, exist_ok=True)
    print(f'{folder_name} created...!')


def get_ignore_labels(ignore_labels: list, class_dict: dict=integer_classes):
    copy_classes = copy.deepcopy(class_dict)
    for class_label in class_dict.keys():
        if int(class_label) in ignore_labels:
            copy_classes.pop(class_label)

    for i, key in enumerate(copy_classes.keys()):
        copy_classes[key] = i

    return copy_classes


def load_files(path: str):
    train_txt_files = glob(path+'train/labels/*.txt')
    valid_txt_files = glob(path+'valid/labels/*.txt')
    test_txt_files = glob(path+'test/labels/*.txt')
    return train_txt_files, valid_txt_files, test_txt_files


def copy_img_folder(origin_path, new_path):
    train_folder = origin_path + 'train/images'
    train_new_folder = new_path + 'train/images'
    copy_tree(train_folder, train_new_folder)

    valid_folder = origin_path + 'valid/images'
    valid_new_folder = new_path + 'valid/images'
    copy_tree(valid_folder, valid_new_folder)

    test_folder = origin_path + 'test/images'
    test_new_folder = new_path + 'test/images'
    copy_tree(test_folder, test_new_folder)



def create_new_txt(folder_name: str, file: str, ignore_class_dict: dict):
    int_ignore_class_dict = {int(k): v for k, v in ignore_class_dict.items()}
    # read existing file
    with open(file, 'r') as f:
        label_list = f.readlines()

    # create new file containing ignore classes
    file_name = file.split('/')
    file_name[3] = folder_name
    new_file_name = '/'.join(file_name)
    new_folder_name = '/'.join(new_file_name.split('/')[:-1])
    
    if not os.path.isdir(new_folder_name):
        os.makedirs(new_folder_name, exist_ok=True)
    
    with open(new_file_name, 'w') as f:
        for label in label_list:
            str_label = label.split(' ')[0]
            bboxes = label.split(' ')[1:]
            if int(str_label) in int_ignore_class_dict.keys():
                # existing label -> new label (converting)
                new_label = ' '.join([str(ignore_class_dict[str_label])] + bboxes)
                f.write(new_label)
            else:
                f.write('')
        print(f'{new_file_name} completed...!')


def create_json_file_for_checking(folder_name: str, ignore_classes_dict: dict, original_classes: dict=inv_classes):
    copy_original_classes = copy.deepcopy(original_classes)
    
    for idx in original_classes.keys():
        if str(idx) not in ignore_classes_dict.keys():
            copy_original_classes.pop(idx)
    
    inv_ignore_class_dict = {v: k for k, v in copy_original_classes.items()}

    count = 0
    for key in inv_ignore_class_dict.keys():
        inv_ignore_class_dict[key] = count
        count += 1

    converted_class_dict = {str(v): k for k, v in inv_ignore_class_dict.items()}

    all_contents = {
        'original_content': inv_classes,
        'converted_content': converted_class_dict,
    }
    
    # create json file
    with open(folder_name+'converted_info.json', 'w', encoding='utf-8') as f:
        json.dump(all_contents, f, indent=4)


def main(args):
    
    # create a new folder for datasets containing ignore classes
    create_new_folder(args.new_folder_name)
    
    # get a dictionary of ignore classes
    ignore_classes_dict = get_ignore_labels(args.ignore_classes)
    
    # load files
    train_files, valid_files, test_files = load_files(args.path)
        
    # convert labels in each dataset files
    new_folder_name = args.new_folder_name.split('/')[-2]
    
    for file in tqdm(train_files):
        create_new_txt(
            folder_name=new_folder_name,
            file=file, 
            ignore_class_dict=ignore_classes_dict,
        )
    print('Complete train set...!')

    for file in tqdm(valid_files):
        create_new_txt(
            folder_name=new_folder_name,
            file=file, 
            ignore_class_dict=ignore_classes_dict,
        )
    print('Complete valid set...!')

    for file in tqdm(test_files):
        create_new_txt(
            folder_name=new_folder_name,
            file=file, 
            ignore_class_dict=ignore_classes_dict,
        )
    print('Complete test set...!')
    
    # create a json file of our converter information
    create_json_file_for_checking(folder_name=args.new_folder_name, ignore_classes_dict=ignore_classes_dict)
    
    # copy image folders
    copy_img_folder(args.path, args.new_folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set Ignore Classes', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
