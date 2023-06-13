import os
import json
import numpy as np
import pandas as pd
import shutil

def json2dataframe(
        json_path: str
) -> pd.DataFrame:
    with open(json_path, 'r') as jfile:
        metadata = json.load(jfile)

    images = pd.DataFrame.from_dict(metadata['images'])
    annotations = pd.DataFrame.from_dict(metadata['annotations'])
    categories = pd.DataFrame.from_dict(metadata['categories'])
    df = pd.merge(
            left=pd.merge(
                    left=images,
                    right=annotations,
                    left_on='id',
                    right_on='image_id',
                    suffixes=['_i', '_a']
            ),
            right=categories,
            left_on='category_id',
            right_on='id'
    )
    return df

def coco2yolo(json_bbox: str) -> pd.DataFrame:
    
    # set up which directories we want things to go into
    out_dir = 'yolov5/images/'
    out_dir2 = 'yolov5/labels/'
    
    # read descriptions
    with open(json_bbox, 'r') as jsonfile:
        metadata = json.load(jsonfile)
    
    # set up variables
    i = 0
    to_txt = ''
    x = 0
    y = 0
    width = 0
    height = 0
    cat = 0
    
    # loop through all given images with descriptions
    while(i < len(metadata['images'])):
        
        # check if there was anything detected, if not move onto next
        if(len(metadata['images'][i]['detections']) != 0):
            
            # image file name
            file = metadata['images'][i]['file']
            
            # change to txt file so we can store bboxes in labels
            to_txt = file.replace('.jpg', '.txt')
            
            # think we have to do this since we don't have test directory
            file = file.removeprefix('test/')
            
            # copy image into test directory in images directory
            shutil.copy(file, out_dir+'test/')
            
            # create file for bboxes in yolo format
            with open(out_dir2+to_txt, 'w') as f:
                
                j = 0
                
                # loop through the detections
                while j < len(metadata['images'][i]['detections']):
                    
                    # set up variables to go in txt file
                    cat = metadata['images'][i]['detections'][j]['category']               
                    x = metadata['images'][i]['detections'][j]['bbox'][0]
                    y = metadata['images'][i]['detections'][j]['bbox'][1]
                    width = metadata['images'][i]['detections'][j]['bbox'][2]
                    height = metadata['images'][i]['detections'][j]['bbox'][3]
    
                    # calculate centers for yolo format
                    x_center = (x + width)/2
                    y_center = (y + height)/2
 
                    # transfer detections to txt file
                    f.write(str(cat) + ' ')
                    f.write(str(x_center) + ' ')
                    f.write(str(y_center) + ' ')
                    f.write(str(width) + ' ')
                    f.write(str(height) + ' ')
                    f.write('\n')
                    
                    j += 1
              
        i += 1
        
    # put into dataframe so we can look at csv file
    image = pd.DataFrame.from_dict(metadata['images'])

    return image

def setupyolo():
    out_dir = 'yolov5/'
    os.makedirs(os.path.join(out_dir, 'images/'))
    os.makedirs(os.path.join(out_dir, 'labels/'))
    
    out_dir2 = 'yolov5/images/'
    out_dir3 = 'yolov5/labels/'
    os.makedirs(os.path.join(out_dir2, 'test/'))
    os.makedirs(os.path.join(out_dir3, 'test/'))
    
    


if __name__ == '__main__':
    # comment this out if running more than once - it creates an error
    setupyolo() 
    
    # create a dataframe from the json annotations file
    anot_df = json2dataframe(
        #json_path='iwildcam2022_train_annotations.json'
        json_path='annotations_train.json'
    )
    anot_df.to_csv('annotations.csv', index=False)
    
    # change coco to yolo format, made into csv to check file
    bbox_df = coco2yolo('iwildcam2022_mdv4_detections.json')
    bbox_df.to_csv('descriptions.csv', index=False)
    
    # tells you how many images there are
    print(anot_df.describe())