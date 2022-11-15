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

- convert voc to yolo format [code](https://github.com/Sangh0/Obstacle-Detection/blob/main/src/preprocessing/convert_voc2yolo.py)

```
$ python3 convert_voc2yolo.py --path ./dataset_dir/
```
  
- convert voc to coco format [code](https://github.com/Sangh0/Obstacle-Detection/blob/main/src/preprocessing/convert_voc2coco.py) 

```
$ python3 convert_voc2coco.py --path ./dataset_dir/
```

- check the balance of classes [code](https://github.com/Sangh0/Obstacle-Detection/blob/main/src/preprocessing/class_count.py)
```
$ python3 class_count.py --path ./dataset/
```
