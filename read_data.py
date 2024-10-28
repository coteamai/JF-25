import pandas as pd
from hyperparams import ddsm_dicom,calc_train,mass_train
from utils import plot_mammography
from pathlib import Path
import os
import re
import cv2
import numpy as np
from typing import Literal,Tuple
import pickle
def read_ddsm_dicom_info(mode:Literal["train","test"]="train"):
    data_raw=pd.read_csv(ddsm_dicom)
    patient_mask=data_raw['PatientID'].str.contains(r'-(Training)_',regex=True)
    match mode:
        case "train":
            patient_mask=data_raw['PatientID'].str.contains(r'-(Training)_',regex=True)
            data_raw = data_raw[patient_mask]
        case "test":
            patient_mask=data_raw['PatientID'].str.contains(r'-(Test)_',regex=True)
            data_raw = data_raw[patient_mask]
        case _:
            raise ValueError("Mode can either be 'train' or 'test'!")
    desc = data_raw.SeriesDescription

    mask_roi=desc=="ROI mask images"
    mask_crop=desc=="cropped images"
    #calc_mask=patient_name.str[:4]=="Calc"
    #mass_mask=patient_name.str[:4]=="Mass"
    

    masked_roi=data_raw[mask_roi]
    masked_crop=data_raw[mask_crop]
    #calc_patients=patient_name[calc_mask]
    #mass_patient=patient_name[mass_mask]

    roi_names=masked_roi.image_path
    crop_names=masked_crop[["image_path","PatientID"]]
    directories={}
    for name in roi_names:
        path_name=str(Path(name).parent)[5:]
        if len(os.listdir(path_name))<2:
            continue
        directories[path_name]=[name[5:]]
    for _,(name_crop,id) in crop_names.iterrows():
        path_name=str(Path(name_crop).parent)[5:]
        if directories.get(path_name):
            directories[path_name].extend([name_crop[5:],id])
    return list(directories.values())
    
def read_bboxes(directory,resize_mask=False,xyxy=True):
    roi,img=directory
    mask = cv2.imread(roi, cv2.IMREAD_GRAYSCALE)
    if resize_mask:
        img=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        mask=cv2.resize(mask,[img.shape[1],img.shape[0]])
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes=[]
    for contour in contours:
        contour=np.array(contour).reshape(-1, 2)
        x_s=contour[:, 0]
        y_s=contour[:, 1]
        x_s, y_s = x_s/mask.shape[1], y_s/mask.shape[0]
        bbox=[x_s.mean(), y_s.mean(), x_s.max() - x_s.min(), y_s.max() - y_s.min()]
        bbox[0]=bbox[0]-bbox[2]/2
        bbox[1]=bbox[1]-bbox[3]/2
        if xyxy:
            bbox[2]=bbox[0]+bbox[2]
            bbox[3]=bbox[1]+bbox[3]
        bboxes.append(bbox)
    return bboxes
def process_directory(directory:Tuple[str,str,str]):
    roi,img,id=directory
    is_mass=id[:4].lower()=="mass"
    assesment_file=mass_train if is_mass else calc_train
    assesment_raw_data=pd.read_csv(assesment_file)
    pattern = r"P_\d{5}"
    patient_id=re.search(pattern,id).group()
    patient=assesment_raw_data[assesment_raw_data.patient_id==patient_id]
    assesment_score=patient.assessment.to_list()[0]
    assesment_score=int(assesment_score)
    bboxes=read_bboxes((roi,img))
    img=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    clahe=cv2.createCLAHE(clipLimit=50)
    img=clahe.apply(img)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img=np.asarray(img)

    return img,(bboxes,assesment_score)
if __name__=="__main__":
    direct=read_ddsm_dicom_info()[1]
    bboxes=read_bboxes((direct[0],direct[1]))
    plot_mammography(direct[1],direct[0],bboxes,resize_mask=True)
    
    #length=[]
    #for direc in read_ddsm_dicom_info(mode="train"):
    #    _,(bbox,_)=process_directory(direc)
    #    length.append(len(bbox))
    #print(length)
    #print(max(length))
    #plot_ddsm()
    #x_sizes=[]
    #y_sizes=[]
    #assesment_scores=[]
    #
    #    assesment_scores.append(score)
    #    x_sizes.append(img.shape[0])
    #    y_sizes.append(img.shape[1])
    #with open("saved/x_sizes.pickle","wb") as x_sizes_f:
    #    pickle.dump(x_sizes,x_sizes_f)
    #with open("saved/y_sizes.pickle","wb") as y_sizes_f:
    #    pickle.dump(y_sizes,y_sizes_f)
    #with open("saved/assesments.pickle","wb") as assesments_f:
    #    pickle.dump(assesment_scores,assesments_f)

    #direct=read_ddsm_dicom_info()[100]
    #bboxes=read_bboxes(direct)
    #plot_mammography(direct[1],direct[0],bboxes)
    #bboxes=read_bboxes(direct,False)
    #plot_mammography(direct[1],direct[0],bboxes,False)
