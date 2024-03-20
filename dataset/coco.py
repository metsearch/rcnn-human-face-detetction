import json
import datetime

import pandas as pd
from rich.progress import track
from utilities.utils import *

import matplotlib.pyplot as plt

date = datetime.date.today()
year, month, day = str(date).split('-')

COCO_dataset_struct = {
    'info':{
        'year': year, 
        'version': '1.0', 
        'description': 'Human faces detection dataset from Kaggle', 
        'contributor': 'Metsearch', 
        'url': 'https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection', 
        'date_created': f'{year}-{month}-{day}'
    },
    'license':[
        {
            'id': 0,
            'name': 'Metsearch', 
            'url': '',
        }
    ],
    'images': [],
    'annotations': [],
    'categories': [{
        'id': 1, 
        'name': 'face', 
        'supercategory': 'human',
    }]
}

def create_coco_dataset(path2data, COCO_dataset=COCO_dataset_struct):
    logger.info('Creating COCO dataset...')
    path2images = os.path.join(path2data, 'images')
    df = pd.read_csv(os.path.join(path2data, 'faces.csv'))
    
    images_paths = pull_images(path2images)
    for i, image_path in track(enumerate(images_paths)):
        image_name = os.path.basename(image_path)
        image_id = image_name.split('.')[0]
        width = int(df[df['image_name'] == image_name]['width'].values[0])
        height = int(df[df['image_name'] == image_name]['height'].values[0])
        if (width != None) and (height != None):
            img_dict = {
                'id': int(image_id),
                'license': COCO_dataset['license'][0]['id'],
                'coco_url': '',
                'flickr_url': '',
                'width': width,
                'height': height,
                'file_name': image_name,
                'date_captured': f'{year}-{month}-{day}'
            }
            COCO_dataset['images'].append(img_dict)
            
            df_image = df[df['image_name'] == image_name]
            for _, row in df_image.iterrows():
                width, height = row['width'], row['height']
                x0, y0 = row['x0'], row['y0']
                x1, y1 = row['x1'], row['y1']
                box_width, box_height = (x1 - x0), (y1 - y0)
                box_area = box_width * box_height
                annot_dict = {
                    'id' : i+1,
                    'category_id': 1,
                    'iscrowd' : 0,
                    'segmentation': [],
                    'image_id' : int(image_id),
                    'area' : box_area,
                    'bbox' : [x0, y0, box_width, box_height]
                }
                COCO_dataset['annotations'].append(annot_dict)
    
    with open('dataset/coco.json', 'w') as f:
        json.dump(COCO_dataset, f)

if __name__ == '__main__':
    logger.info('... [ Testing CocoDataset ] ...')
    path2data = '/home/muhammet/Challenges/00_Datasets/human-faces'
    create_coco_dataset(path2data)