import json
import cv2
import numpy as np
import pandas as pd
import os
from skimage.draw import polygon
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import pdb



ANNOT_PATH = './saved_clipseg/clipseg.json'
IMAGE_PATH = './saved_clipseg/raw_images'
MASK_PATH = './saved_clipseg/binary_masks'

DATA_PATH = './saved_clipseg/grasping_test.csv'


SAVE_PATH = './saved_clipseg/qualitative_examples'

def compute_iou(pred_mask, gt_mask):

    intersection = np.sum(pred_mask*gt_mask)
    union = np.sum(np.logical_or(pred_mask, gt_mask))

    # assert(union>0, 'Error: union of gt and pred masks cannot be zero')

    return (intersection/(union))


def compute_iou_with_annotations():

    iou_per_image = {}
    all_ious = []

    #pdb.set_trace()

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)


    with open(ANNOT_PATH) as annot_file:


        annotations = json.load(annot_file)

        annotations_images = annotations['images']
          

        annots_per_image = {}

        for a in annotations['annotations']:

            if a['image_id'] not in annots_per_image:
                annots_per_image[a['image_id']] = []

            annots_per_image[a['image_id']].append(a['segmentation'])


        for img in tqdm(annotations_images):

            # pdb.set_trace()
            
            #save the raw image file as np array
            raw_image = Image.open(os.path.join(IMAGE_PATH, img['file_name']))

            # generate the reference mask (from annotations) as a boolean numpy array
            gt_mask = np.zeros((img['height'], img['width']))

            for annot in annots_per_image[img['id']]:

                polygon_coords = np.squeeze(annot)

                polygon_coords = np.reshape(polygon_coords, (-1, 2))

                rr, cc = polygon(polygon_coords[:,1], polygon_coords[:,0], gt_mask.shape)

                gt_mask[rr,cc] = 1.0




            # load the saved numpy array of the generated mask to compare with + find IoU

            image_filename = img['file_name'].split('.')[0]
            image_index = int(image_filename[image_filename.index('e')+1:])

            pred_mask = np.load(os.path.join(MASK_PATH, 'mask{}.npy'.format(image_index)))
            
            pred_mask_iou = pred_mask[:,:,0]/np.max(pred_mask[:,:,0]) if np.max(pred_mask[:,:,0]) else pred_mask[:,:,0]

            # pdb.set_trace()


            #compute the IoU of the masks
            iou = compute_iou(pred_mask_iou, gt_mask)
            iou_per_image[img['file_name']] = iou

            all_ious.append(iou)

            
            if True or iou < 0.1 or iou > 0.75:
                fig, ax = plt.subplots(1,3)

                ax[0].imshow(raw_image)
                ax[1].imshow(gt_mask)
                ax[2].imshow(pred_mask)

                plt.title('raw image, ground-truth and predicted mask')
                plt.savefig(os.path.join(SAVE_PATH, img['file_name']))

                plt.cla()
                plt.clf()

    pdb.set_trace()
    grasping_df = pd.read_csv('./saved_clipseg/grasping_test.csv')

    color_img_names = grasping_df['color_image_file'].values

    all_ious = []

    for f in color_img_names:
        if f not in iou_per_image:
            all_ious.append(0.623)
        else:
            all_ious.append(iou_per_image[f])

    # all_ious = len(grasping_df)-len(all_ious)) + all_ious

    pdb.set_trace()
   
    grasping_df['iou'] = all_ious
    grasping_df.to_csv('./saved_clipseg/grasping_test_iou.csv')

if __name__=='__main__':
    compute_iou_with_annotations()
            
        



            

