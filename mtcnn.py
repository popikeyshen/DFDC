

import cv2
import torch
from facerec import FaceRecognizer, image_to_tensor, PersonType, PersonInfo
from utils.images import hconcat_resize_min, get_int_rect

import imutils
import matplotlib.pyplot as plt

#/media/popikeyshen/30c5a789-895a-4cc2-910a-3c678cc563d7/deepfake/train_sample_videos/afoovlsmtx.mp4
#/media/popikeyshen/30c5a789-895a-4cc2-910a-3c678cc563d7/deepfake/train_sample_videos/alninxcyhg.mp4
#/media/popikeyshen/30c5a789-895a-4cc2-910a-3c678cc563d7/deepfake/train_sample_videos/alvgwypubw.mp4


#im = cv2.imread("/media/popikeyshen/30c5a789-895a-4cc2-910a-3c678cc563d7/mtcnn_torch/2020-01-22-143629.jpg.jpg", cv2.IMREAD_COLOR)
#cv2.imshow("im",im)
#cv2.waitKey(0)


device = torch.device('cuda:0')
distance_threshold = 1.0
recognizer = FaceRecognizer(device=device, distance_threshold=distance_threshold)


cap = cv2.VideoCapture("/media/popikeyshen/30c5a789-895a-4cc2-910a-3c678cc563d7/deepfake/train_sample_videos/alvgwypubw.mp4")

while(1):

	ret, im = cap.read()
	im = imutils.resize(im, height=800)

	embeddings, boxes = recognizer.get_faces_from_frame(image_to_tensor(im))
	for box in  boxes:
			#dist, person = recognizer.classify_person(embedding)
			#if person is not None: 
			min_x, min_y, max_x, max_y = get_int_rect(box)
			cv2.rectangle(im, (min_x, min_y), (max_x, max_y), (0,255,0), 1)

			cropped_face = im[ min_y:max_y, min_x:max_x]

			cv2.imshow("cropped_face",cropped_face)
			k = cv2.waitKey(1)

			histr = cv2.calcHist([cropped_face],[0],None,[256],[0,256]) 
  
			# show the plotting graph of an image 
			plt.plot(histr) 
			plt.show() 
			

	cv2.imshow("im",im)
	k = cv2.waitKey(0)

	if k == 'q':
		break
