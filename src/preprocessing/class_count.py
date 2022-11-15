import argparse
import xml.etree.ElementTree as ET
from glob import glob
from tqdm.auto import tqdm

def class_count(path, class_count_dict):
    xml_list = glob(path+'**/**/*.xml')

    for xml in tqdm(xml_list):
        annotations = ET.parse(xml).getroot()
        image_list = annotations.findall('image')

        for image in image_list:
            for label in image:
                class_name = label.get('label')
                class_count_dict[class_name] += 1

    return class_count_dic


def main():
    parser = argparse.ArgumentParser(description='Count number of classes in our dataset')
    parser.add_argument('--path', type=str, required=True,
                        help='dataset directory')
    args = parser.parse_args()
    
    print(class_count(args.path))


if __name__ == "__main__":
    main()
