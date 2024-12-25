from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np

from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2




class ClipSeg:

    def __init__(self, threshold_mask: bool = True, threshold_value: float = 0.5, depth_product: bool = False):
        self.use_mask_thresholding = threshold_mask
        self.mask_threshold = threshold_value
        self.multiply_depth = depth_product

        #load the ClipSeg procdataessor and model
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    def resize_image(self, image: np.ndarray, dimensions: (int, int)):
        """Resize an image to the given dimensions using linear interpolation."""
        return cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)

    def overlay_image(self, image: np.ndarray, mask: np.ndarray, alpha: float):
        """Overlay the foreground image onto the background with a given opacity (alpha)."""
        return cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

    def apply_colormap(self, mask: torch.Tensor, colormap):
        """Apply a colormap to a tensor and convert it to a numpy array."""
        colored_mask = colormap(mask.numpy())[:, :, :3]
        return (colored_mask * 255).astype(np.uint8)

    def segment_image(self, raw_image: np.ndarray, depth_image: np.ndarray, text_prompt: str):

        """
        Generate a segmenetation mask using ClipSeg from the input image for the text_prompt


        Inputs:
        raw_image: input image from SPOT pinhole camera
        text_prompt: input (parsed) language prompt for model to segment
        segment_threshold: if thresholded (binary) mask required, then generate binary mask with the above threshold
        on normalized clip scores
        only_thresholded_mask: boolean representing if only the boolean mask should be returned
        
        Outputs:
        generated_mask: floating point CLIP values assigned to the generated mask
        binary_mask: 0 or 1 values (boolean) assigned to thresholded generated mask
        """

        #convert numpy image array into PIL image
        image = Image.fromarray(raw_image, mode='RGB')


        #preprocess inputs and run through the model
        preprocessed_inputs = self.clipseg_processor(text=[text_prompt], images=image, padding="max_length", return_tensors="pt")

        with torch.no_grad():
            outputs = self.clipseg_model(**preprocessed_inputs)
        
        
        #normalize the generated mask using sigmoid and threshold the mask if necessary
        generated_mask = torch.sigmoid(outputs[0])
        
        
        dimensions = (raw_image.shape[1], raw_image.shape[0])


        #apply thresholding to mask
        thresholded_mask = torch.where(generated_mask >= self.mask_threshold, 1.0, 0.0)

        #convert thresholded mask to a binary mask and resize to original raw image size
        binary_mask = self.apply_colormap(thresholded_mask, cm.Greys_r)
        binary_mask = self.resize_image(binary_mask, dimensions)
        
        #resize the generated mask (clipseg floating point scores) to original raw image size
        generated_mask = torch.from_numpy(self.resize_image(generated_mask.numpy(), dimensions))

        #multiply the generated mask by the depth visual data: invert so that low depth is preferred
        if self.multiply_depth and np.max(depth_image)!=0:

            
            depth_image =  1-depth_image/np.max(depth_image)
            
            # depth_image = 1-torch.sigmoid(torch.from_numpy(depth_image)).numpy()
            generated_mask = torch.mul(generated_mask, depth_image) 


        return generated_mask, binary_mask

        