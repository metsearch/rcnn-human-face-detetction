import os
from glob import glob
from .log import logger

def pull_images(path2files, extension='jpg'):
    return glob(f'{path2files}/*.{extension}')

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    logger.info('Testing utils...')
    files = pull_images('/home/muhammet/Challenges/00_Datasets/drones/images')
    print(len(files))