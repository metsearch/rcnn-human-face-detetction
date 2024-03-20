from PIL import Image
import torch as th
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from utilities.utils import *

class HumanFacesDataset(Dataset):
    def __init__(self, imgs_source, annotation_file, transforms=None):
        super().__init__()
        self.source = imgs_source
        self.coco = COCO(annotation_file)
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))
    
    def __len__(self):
        return len(self.ids)
    
    def _load_image(self, idx):
        image_id = self.ids[idx]
        image_name = self.coco.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(self.source, image_name)
        image = Image.open(image_path)
        return image
    
    def _get_annotation(self, idx):
        image_id = self.ids[idx]
        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        annots = self.coco.loadAnns(annot_ids)
        
        num_objs = len(annots)
        boxes, areas = [], []
        for annot in annots:
            x_min = annot['bbox'][0]
            y_min = annot['bbox'][1]
            x_max = x_min + annot['bbox'][2]
            y_max = y_min + annot['bbox'][3]
            boxes.append([x_min, y_min, x_max, y_max])
            areas.append(annot['area'])
            
        annotation = {
            'boxes': th.as_tensor(boxes, dtype=th.float32),
            'labels': th.ones((num_objs,), dtype=th.int64),
            'image_id': th.tensor([image_id]),
            'areas': th.as_tensor(areas, dtype=th.float32),
            'iscrowd': th.zeros((num_objs,), dtype=th.float64)
        }
        
        return annotation
    
    def __getitem__(self, idx):
        image = self._load_image(idx)
        annotation = self._get_annotation(idx)
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        return image, annotation

if __name__ == '__main__':
    logger.info('... [ Testing dataset ] ...')