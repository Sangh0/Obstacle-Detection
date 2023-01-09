### The codes for preprocessing in my project
- Our directory type is as following:
```
path: ./obstacle_detection/dataset/
├── Bbox_01
│    ├─ Bbox_0001
│       ├─ 0001.json
│	├─ img01.jpg
│       ├─ img02.jpg
│       ├─ ...
│       ├─ img99.jpg
│    ├─ Bbox_0002
│       ├─ 0002.json
│       ├─ img01.jpg
│       ├─ img02.jpg
│       ├─ ...
│       ├─ img99.jpg
│    ├─ ...
├── Bbox_02
│    ├─ Bbox_0100
│       ├─ 0100.json
│       ├─ img01.jpg
│       ├─ ...
│       ├─ img99.jpg
│    ├─ Bbox_0101
│       ├─ 0101.json
│       ├─ img01.jpg
│       ├─ ...
│       ├─ img99.jpg
├── ...
```

**VOC to YOLO format converter** [code](https://github.com/Sangh0/Obstacle-Detection/blob/main/src/preprocessing/voc2yolo_converter.py)

```
$ python3 voc2yolo_converter.py --path {data directory}
```
  
**VOC to COCO format converter** [code](https://github.com/Sangh0/Obstacle-Detection/blob/main/src/preprocessing/voc2coco_converter.py) 

```
$ python3 voc2coco_converter.py --path {data directory}
```

**check the balance of classes** [code](https://github.com/Sangh0/Obstacle-Detection/blob/main/src/preprocessing/class_count.py)
```
$ python3 class_count.py --folder_category {'large' or 'detail'} --path {data directory} --save_file_name {csv file name}
```

**Generate new labels with ignore classes**
```
$ python3 create_ignore_class.py --path {original directory} --new_folder_name {new folder name} --ignore_classes {a list containing of ignore classes}
```
