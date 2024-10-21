import pandas as pd
from hyperparams import ddsm_dicom
from utils import plot_mammography
from pathlib import Path
import os
import cv2
import numpy as np
def read_ddsm_dicom_info():
    data_raw=pd.read_csv(ddsm_dicom)
    desc=data_raw.SeriesDescription
    mask_roi=desc=="ROI mask images"
    mask_crop=desc=="cropped images"
    masked_roi=data_raw[mask_roi]
    masked_crop=data_raw[mask_crop]
    roi_names=masked_roi.image_path
    crop_names=masked_crop.image_path
    directories={}
    for name in roi_names:
        path_name=str(Path(name).parent)[5:]
        if len(os.listdir(path_name))<2:
            continue
        directories[path_name]=[name[5:]]
    for name_crop in crop_names:
        path_name=str(Path(name_crop).parent)[5:]
        if directories.get(path_name):
            directories[path_name].append(name_crop[5:])
    return list(directories.values())
    
def read_bboxes(directory,resize_mask=True):
    roi,img=directory
    img=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(roi, cv2.IMREAD_GRAYSCALE)
    if resize_mask:
        mask=cv2.resize(mask,[img.shape[1],img.shape[0]])
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes=[]
    for contour in contours:
        contour=np.array(contour).reshape(-1, 2)
        x_s=contour[:, 0]
        y_s=contour[:, 1]
        x_s, y_s = x_s/mask.shape[1], y_s/mask.shape[0]
        bbox=(x_s.mean(), y_s.mean(), x_s.max() - x_s.min(), y_s.max() - y_s.min())
        bboxes.append(bbox)
    return bboxes
def process_directory(directory):
    roi,img=directory
    bboxes=read_bboxes((roi,img))
    img=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    clahe=cv2.createCLAHE(clipLimit=50)
    img=clahe.apply(img)
    img=np.asarray(img)
    return img,bboxes
if __name__=="__main__":
    direct=read_ddsm_dicom_info()[100]
    bboxes=read_bboxes(direct)
    plot_mammography(direct[1],direct[0],bboxes)
    bboxes=read_bboxes(direct,False)
    plot_mammography(direct[1],direct[0],bboxes,False)
