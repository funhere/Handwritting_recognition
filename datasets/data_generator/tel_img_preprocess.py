"""
trainig,validation,test imageに前処理を実施するスクリプト
1). resize: 48x145 -> 416x416;
2). -gaussian-blur;
3). COLOR_RGB2GRAY
4). update xml file
"""

import os
import cv2
import numpy as np
from shutil import copyfile
import subprocess
import xml.etree.ElementTree as etree
from xml.etree.ElementTree import ParseError

path_list = [
    # train
     '/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/left/02-01'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/left/02-02'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/left/02-07'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/left/02-08'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/left/02-19'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/mid/02-08'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/mid/02-09'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/mid/02-19'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/right/02-13'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/right/02-14'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/01_train/right/02-19'
    # val
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/02_val/left/02-19'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/02_val/mid/02-19'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/02_val/mid/02-20'
    ,'/Users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/AI/submission/02_val/right/02-20'
    # test
    ,'/users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/submission/03_test/left/02-21'
    ,'/users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/submission/03_test/left/03-02'
    ,'/users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/submission/03_test/left/03-05'
    ,'/users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/submission/03_test/mid/02-22'
    ,'/users/shiraishihiroki/OneDrive - SMFLキャピタル株式会社/submission/03_test/right/02-23'
]

idx = 0
for path in path_list:
    xml_list = [file for file in os.listdir(path) if file.endswith('.xml')]
    jpg_list = [file for file in os.listdir(path) if file.endswith('.jpg')]

    for file in xml_list:
        img_name = file.split('.xml')[0]+'.jpg'
        img_path = os.path.join(path,img_name)
        assert img_name in jpg_list
        img = cv2.imread(img_path)
        img = np.vstack([np.zeros(shape=(48,145,3)),img,np.zeros(shape=(49,145,3))])
        #cv2.imwrite(os.path.join('images',str(idx)+'.jpg'),img)
        cv2.imwrite(os.path.join('images',img_name),img)
        #copyfile(os.path.join(path,file), os.path.join('annotations','xmls',str(idx)+'.xml'))
        copyfile(os.path.join(path,file), os.path.join('annotations','xmls',file))
        try:
            #tree = etree.parse(os.path.join('annotations','xmls',str(idx)+'.xml'))
            tree = etree.parse(os.path.join('annotations','xmls',file))
            root = tree.getroot()
            root[4][2].text = '3'
            root[1].text = str(idx)+'.jpg'
            for l in range(len(root)):
                if root[l].tag == 'object':
                    root[l][4][1].text = str(int(root[l][4][1].text)+48)
                    root[l][4][3].text = str(int(root[l][4][3].text)+48)
            #tree.write(os.path.join('annotations','xmls',str(idx)+'.xml'))
            tree.write(os.path.join('annotations','xmls',file))
        except ParseError:
            print ("Error",file)
        idx += 1

path = 'images'
jpg_list = [file for file in os.listdir(path) if file.endswith('.jpg')]

# 2値化の変数
MAXVALUE  = 255


for file in jpg_list:
    command = ['convert',os.path.join(path,file),'-quality','100','-resize','416x416!',os.path.join(path,file)]
    subprocess.call(command)
    command = ['convert',os.path.join(path,file),'-gaussian-blur','1x8',os.path.join(path,file)]
    subprocess.call(command)
    img = cv2.imread(os.path.join(path,file))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    mat = (img > 2) #true=1, false=0
    ave=np.sum(img*mat)/np.sum(mat)
    THRESHOLD = ave
    MAXVALUE = 255
    _, img = cv2.threshold(img, THRESHOLD, MAXVALUE, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(path,file),img)
    command = ['convert',os.path.join(path,file),'-gaussian-blur','1x1',os.path.join(path,file)]
    subprocess.call(command)
path = 'annotations/xmls'
xml_list = [file for file in os.listdir(path) if file.endswith('.xml')]
for file in xml_list:
    try:
        tree = etree.parse(os.path.join(path,file))
        root = tree.getroot()
        root[4][0].text = '416'
        root[4][1].text = '416'
        for l in range(len(root)):
            if root[l].tag == 'object':
                xmin = int(root[l][4][0].text)
                ymin = int(root[l][4][1].text)
                xmax = int(root[l][4][2].text)
                ymax = int(root[l][4][3].text)
                xmin = int(xmin*(416/145))
                ymin = int(ymin*(416/145))
                xmax = int(xmax*(416/145))
                ymax = int(ymax*(416/145))
                root[l][4][0].text = str(xmin)
                root[l][4][1].text = str(ymin)
                root[l][4][2].text = str(xmax)
                root[l][4][3].text = str(ymax)
        tree.write(os.path.join(path,file))
    except ParseError:
        print ("Error",file)
