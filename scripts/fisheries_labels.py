import xml.etree.ElementTree as ET
import pickle
import os
from glob import glob
import json
from PIL import Image
import shutil
import numpy as np

labels = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
DATADIR = '/media/arun/data/kaggle/the-nature-conservancy-fisheries-monitoring'


def main():

	# Clean up dataset.
	print('Cleaning up directories')
	shutil.rmtree(os.path.join(DATADIR,'dataset','train'))
	shutil.rmtree(os.path.join(DATADIR,'dataset','labels'))


	print('Creating directories..')
	os.makedirs(os.path.join(DATADIR,'dataset','train'))
	os.makedirs(os.path.join(DATADIR,'dataset','labels'))
	
	train = []
	for label in labels:
		annotations_file = os.path.join(DATADIR,'bbox',label.lower()+'_labels.json')
		with open(annotations_file,'r') as f:
			annotations = json.load(f)

		for annotation in annotations:

			img_file = os.path.basename(annotation['filename'])
			img_path = os.path.join(DATADIR,'train',label,img_file)
			im = Image.open(img_path)

			size = im.size
			dw = 1./size[0]
			dh = 1./size[1]

			bbox = []
			for a in annotation['annotations']:
				
				w = a['width']
				h = a['height']

				x = a['x'] + w/2.0
				y = a['y'] + h/2.0
				
				x = x*dw
				y = y*dh
				w = w*dw
				h = h*dh

				bbox.append([labels.index(label),x,y,w,h])

			if len(bbox) > 0:

				# Copy image to the train folder
				tgt_img_path = os.path.join(DATADIR,'dataset','train',img_file)
				shutil.copy(img_path,tgt_img_path)
				train.append(tgt_img_path)			

				# Save bboxes to the labels folder
				bbox_file = os.path.join(DATADIR,'dataset','train',img_file.replace('.jpg','.txt'))
				np.savetxt(bbox_file,np.array(bbox),fmt="%d %f %f %f %f")



	np.savetxt(os.path.join(DATADIR,'dataset','train.txt'),train,'%s')

	# Create a test file.
	test = []
	for img in glob(os.path.join(DATADIR,'test_stg1','img*.jpg')):
		test.append(img)

	np.savetxt(os.path.join(DATADIR,'dataset','test.txt'),test,'%s')


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
