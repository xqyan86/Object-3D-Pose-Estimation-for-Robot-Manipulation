#!/usr/bin/env python

# USE_GAZEBO = True

# Import required Python code.
import math
from matplotlib import pyplot as plt
import os
import csv
import cPickle

#---- ROS ------
import roslib; roslib.load_manifest('deep_pose_estimation')
import rospkg
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import actionlib
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import *
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
# ---- For Gazebo -----
import std_srvs.srv as srv
from geometry_msgs.msg import PoseArray
import yaml
import caffe
from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
import argparse
import sys
import time
import numpy as np
from stl import mesh
import cv2
import cv_bridge


MAX_DEPTH_RANGE = 7000.0 #mm
#SAVE_CONTINUOUS_DATA = False
DRAW_EACH_IMAGE_FEATURE = False
#ROS_RUN_RATE = 15
ROS_RUN_RATE = 1
MAX_NUMBER_IMAGES = ROS_RUN_RATE*60 # 1 min

SAVE_IMAGE = True

CLASSES = ('__background__', # always index 0
			'002_master_chef_can',
			'003_cracker_box',
			'004_sugar_box',
			'005_tomato_soup_can',
			'006_mustard_bottle',
			'007_tuna_fish_can',
			'008_pudding_box',
			'009_gelatin_box',
			'010_potted_meat_can',
			'011_banana',
			'019_pitcher_base',
			'021_bleach_cleanser',
			'024_bowl',
			'025_mug',
			'035_power_drill',
			'036_wood_block',
			'037_scissors',
			'040_large_marker',
			'051_large_clamp',
			'052_extra_large_clamp',
			'061_foam_brick',
			)

CLASS_TO_IND = dict(zip(CLASSES, range(len(CLASSES))))

COLORS = np.random.random((len(CLASSES), 3)) * 255
COLORS = COLORS.astype(np.int)

NNFile_path = '/home/xiaoqiangyan/Documents/py-faster-rcnn-pose-allregress/'
NETS = {'vgg16': ('VGG16',
		NNFile_path+'output/faster_rcnn_end2end/ycb__train/vgg16_faster_rcnn_pose_iter_50000.caffemodel')}

CONF_THRESH = 0.2
NMS_THRESH = 0.5


class PerceptionParameterNode():
	def __init__(self,name):
		self.bridge = CvBridge()

		self.img_counter = 0
		self.rgb_counter = 0
		self.save_rgb = 0
		self.depth_counter = 0
		self.start_saving_continuous_data = False
		self.continuous_data_directory_name = ""

		self.have_image  = False
		self.have_depth_image = False
		self.rgb_image   = 0

		#====== Subscriber ======
		#print("USE_GAZEBO", self.USE_GAZEBO)
		#if self.USE_GAZEBO:
		#    image_topic = "/asus/rgb/image_rect_color"
		#    depth_image_topic = "/asus/depth/image_rect_raw"
		#else :
		image_topic = "/asus/rgb/image_raw" # This works with real asus
		#image_topic = "/dataset_image"
		#dataset_image_topic = "/dataset_image"
		#self.pub = rospy.Publisher(dataset_image_topic, Image, latch=True)
		#image_topic = "/asus/rgb/image_rect_color"
		# depth_image_topic = "/asus/depth/image_raw"  # This works with real asusimport cPickle
		
		#================ Initialize Marker=====================
		marker_topic = "/visualization_marker_array"
		self.publisher = rospy.Publisher(marker_topic, MarkerArray)
		#self.markerArray = MarkerArray()
		   # initial starting location I might want to move to the param list
		# self.robotMarker = Marker()
		#self.markerArray = MarkerArray()
		# self.robotMarker.header.frame_id = "kinect_optical_frame"
		# #self.robotMarker.type = marker.SPHERE
		# #self.robotMarker.action = marker.ADD
		# self.h = rospy.get_param("height", 100)
		# self.w = rospy.get_param("width", 100)
		# # self.state.point.x = self.h
		# # self.state.point.y = self.w
		# # self.state.point.z = 0

		# self.robotMarker.header.stamp    = rospy.get_rostime()
		# self.robotMarker.ns = "Detection"
		# self.robotMarker.id = 0
		# self.robotMarker.type = 3 # cubes
		# self.robotMarker.action = 0
		# self.robotMarker.pose.position.x = 1
		# self.robotMarker.pose.position.y = 1
		# self.robotMarker.pose.position.z = 1 
		# self.robotMarker.pose.orientation.x = 0
		# self.robotMarker.pose.orientation.y = 0
		# self.robotMarker.pose.orientation.z = 0
		# self.robotMarker.pose.orientation.w = 1.0
		# self.robotMarker.scale.x = 0.1
		# self.robotMarker.scale.y = 0.1
		# self.robotMarker.scale.z = 0.1

		# self.robotMarker.color.r = 0.0
		# self.robotMarker.color.g = 1.0
		# self.robotMarker.color.b = 0.0
		# self.robotMarker.color.a = 1.0

		#self.robotMarker.lifetime = rospy.Duration(0)


		cfg.TEST.HAS_RPN = True  # Use RPN for proposals
		cfg.TEST.SCALES = (600, )
		cfg.TEST.MAX_SIZE = 1333

		args = self.parse_args()

		prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
							'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
		caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
							  NETS[args.demo_net][1])

		prototxt = NNFile_path+'models/ycb/VGG16/faster_rcnn_end2end/test.prototxt'
		caffemodel = NETS[args.demo_net][1]

		if not os.path.isfile(caffemodel):
			raise IOError(('{:s} not found.\nDid you run'+ NNFile_path + 'data/script/'
				   'fetch_faster_rcnn_models.sh?').format(caffemodel))

		if args.cpu_mode:
			caffe.set_mode_cpu()
		else:
			caffe.set_mode_gpu()
			caffe.set_device(args.gpu_id)
			cfg.GPU_ID = args.gpu_id

		self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

		print '\n\nLoaded network {:s}'.format(caffemodel)

		self.image_sub       = rospy.Subscriber(image_topic,Image,self.image_callback) #/asus or camera
		# self.depth_image_sub = rospy.Subscriber(depth_image_topic, Image, self.depth_image_callback)

		rospy.loginfo('done')

	def draw_detections(self, im, dets, thresh=0.5):
		"""Draw detected bounding boxes."""
		show_im = im.copy()
		keep_inds = np.where(dets[:, -2] >= thresh)[0]
		dets = dets[keep_inds,:]
		if len(keep_inds) == 0:
			return show_im
		#del_indexes = []
		print ('suprise!!')
		print len(keep_inds), dets.shape
		i = 0
		while (i < dets.shape[0]):
			cls_ind = int(dets[i, -1])
		
			bbox = dets[i, :4]
			score = dets[i, -2]
			x1, y1, x2, y2 = bbox
			
			cv2.rectangle(show_im, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[cls_ind, :], 3)
			font=cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(show_im, CLASSES[cls_ind], (x1,y1), font, 0.5, (255,255,255),2)
			
			i += 1
			#print dets.shape[0], c.shap
			
		return show_im

	#accuracy_pose = np.zeros((22,2),dtype=np.float32)

	def yaml_loader(self, filepath):
		"""load yaml file"""
		with open(filepath, "r") as file_descriptor:
			data = yaml.load(file_descriptor)
		return data
	
	def parse_args(self):
		"""Parse input arguments."""
		parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
		parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
						default=0, type=int)
		parser.add_argument('--cpu', dest='cpu_mode',
						help='Use CPU mode (overrides --gpu)',
						action='store_true')
		parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
						choices=NETS.keys(), default='vgg16')

		args = parser.parse_args()

		return args
	
	def pose_tranfer(self, rot):
		"""tranfer the rotation from quanternion to matrix"""
		q0, q1, q2, q3 = rot[i,:]
		R0 = [q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2]
		R1 = [2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1]
		R2 = [2*q1*q3-2*q0*q2, 2*q2*q3-2*q0*q1, q0**2-q1**2-q2**2+q3**2]
		rot = np.vstack((R0,R1,R2))
		return rot

	def image_callback(self,data):
		try:
			self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			self.have_image = True
			t = rospy.Time.now()
			self.rgb_counter = self.rgb_counter + 1
			
			#if self.rgb_counter == 10:
			self.save_rgb +=1
			#image_rgb_name = "/home/xiaoqiangyan/Documents/MyData/RGBImage/"+str(self.save_rgb) + ".png"
			#cv2.imwrite(image_rgb_name, self.rgb_image)
		#image_rgb_name = "time_str(self.rgb_counter).jpg"  time = ros time

		#TODO your code here
		#get cam_matrix fromo .yaml file
		#camera_intrinsic = self.yaml_loader("~/work/catkin_ws/src/deep_pose_estimation/rgb_PS1080_PrimeSense.yaml")
		#cam_matrix = camera_intrinsic['projection_matrix']
		#print cam_matrix
		#principal = np.hstack((cam_matrix[0,2], cam_matrix[1,2]))
		#focal_len = np.hstack((cam_matrix[0,0],cam_matrix[1,1]))
		
			print ('******* Image ' + str(self.save_rgb) + ' ***********')
			# # Detect all object classes and regress object bounds
			start = time.time()
			#im = cv2.imread('/data/YCB-dataset/YCB_Video_Dataset/data/0001/000001-color.png')
			#msg = cv_bridge.CvBridge().cv2_to_imgmsg(im, encoding="bgr8")
			#msg.header.frame_id = "kinect_optical_frame"
			#msg.header.stamp = rospy.get_rostime()
			#pub = rospy.Publisher(image_topic, Image, latch=True)
			#self.pub.publish(msg)
			im = self.rgb_image.copy()
			assert im is not None
			w = np.size(im,1)
			h = np.size(im,0)
			#print (w,h)

			#crop_im = im[h/4:3/4*h, w/4:3/4*w]
			resize_im = cv2.resize(im,(w,960))
			crop_im = resize_im[480:960,320:960]
			cv2.imshow("resized",resize_im)
			cv2.waitKey(0)
			w1 = np.size(crop_im,1)
			h1 = np.size(crop_im,0) 
			print (w1,h1)
			#cv2.imshow("cropped", crop_im)
			#cv2.waitKey(0)
			show_im = crop_im.copy()
			#we need to set the gpu mode in this function
			caffe.set_mode_gpu()
			caffe.set_device(0)
			#caffe.set_device(args.gpu_id)
			#print (args)
			scores, boxes, trans, rot = im_detect(self.net, crop_im)
			# print(scores.shape, boxes.shape, tz.shape, rot.shape)
			end = time.time()
			#print ('Detection took {:.3f}s for '
					#'{:d} object proposals').format((end-start), boxes.shape[0])
			print ('Detection took {:.3f}s'.format(end-start))

			all_dets = np.zeros((0, 6), dtype=np.float32)
			all_trans = np.zeros((0, 3), dtype=np.float32)
			all_rot = np.zeros((0, 4), dtype=np.float32)

			for cls_ind, cls in enumerate(CLASSES[1:]):
				#print('cls'.format(cls))
				cls_ind += 1 # because we skipped background
				cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
				cls_scores = scores[:, cls_ind]
				dets = np.hstack((cls_boxes,
								  cls_scores[:, np.newaxis])).astype(np.float32)
				keep = nms(dets, NMS_THRESH)
				dets = dets[keep, :]
				# keep = np.where(dets[:, -1] >= CONF_THRESH)[0]
				# dets = dets[keep, :]
				# print(np.min(dets[:, -1]), np.max(dets[:, -1]))
				dets = np.hstack((dets,
								  cls_ind * np.ones((dets.shape[0], 1), dtype=np.float32)
								  ))

				all_dets = np.vstack((all_dets, dets))
				all_trans = np.vstack((all_trans, trans[keep, :]))
				all_rot = np.vstack((all_rot, rot[keep, :]))
				

			
			show_im = self.draw_detections(show_im, all_dets, thresh=CONF_THRESH)
			#pick the proposals that scores are bigger than CONF_THRESH
			keep = np.where(all_dets[:, -2] >= CONF_THRESH)[0]
			all_dets = all_dets[keep, :]
			all_trans = all_trans[keep, :]    ###here because the focus length is different, so /2
			all_rot = all_rot[keep, :]

			focal_len = [1177.80517578125, 1196.55615234375]
			principal = [727.6066148836981, 528.2544818723254]
			cx = all_trans[:,0] * focal_len[0]/all_trans[:,2] + principal[0]-320
			cy = all_trans[:,1] * focal_len[1]/all_trans[:,2] + principal[1]-480
			cx = cx[:,np.newaxis]
			cy = cy[:,np.newaxis]
			#center = np.hstack((cx,cy))

			if all_trans.shape[0] != 0:
				
				print('Objects Detected and the Predicted 3D Poses: ')
				
				markerArray = MarkerArray()
				for idx in range(all_trans.shape[0]):
					cls_ind = int(all_dets[idx, -1])
					print (cx[idx],cy[idx])
					cv2.circle(show_im,(int(cx[idx]),int(cy[idx])),1,(255,215,0),3)
					# cv2.imshow('circle',show_im)
					# cv2.waitKey(0)

					robotMarker = Marker()
					robotMarker.header.frame_id = "kinect_optical_frame"
					h = rospy.get_param("height", 100)
					w = rospy.get_param("width", 100)
					# self.state.point.x = self.h
					# self.state.point.y = self.w
					# self.state.point.z = 0

					robotMarker.header.stamp    = rospy.get_rostime()
					robotMarker.ns = "Detection"
					robotMarker.id = 0
					#robotMarker.type = 3 # cubes
					robotMarker.type = Marker.MESH_RESOURCE
					robotMarker.mesh_use_embedded_materials = True
					robotMarker.mesh_resource = "package://deep_pose_estimation/009_gelatin_box/textured.obj"

					robotMarker.action = 0

					robotMarker.pose.position.x = all_trans[idx,0]
					robotMarker.pose.position.y = all_trans[idx,1]
					robotMarker.pose.position.z = all_trans[idx,2]

					robotMarker.pose.orientation.x = all_rot[idx,0]
					robotMarker.pose.orientation.y = all_rot[idx,1]
					robotMarker.pose.orientation.z = all_rot[idx,2]
					robotMarker.pose.orientation.w = all_rot[idx,3]
					# robotMarker.pose.orientation.x = 0.6402
					# robotMarker.pose.orientation.y = 0.3852
					# robotMarker.pose.orientation.z = 0.1357
					# robotMarker.pose.orientation.w = 0.651
					# robotMarker.pose.orientation.x = 0.3852
					# robotMarker.pose.orientation.y = 0.6402
					# robotMarker.pose.orientation.z = 0.1357
					# robotMarker.pose.orientation.w = 0.651
					robotMarker.id = idx

					robotMarker.scale.x = 2
					robotMarker.scale.y = 2
					robotMarker.scale.z = 2

					#print (COLORS[cls_ind,:])
					# robotMarker.color.r = float((idx+1)/10.0) *(idx+1)**2   
					# robotMarker.color.g = float((idx+1)/10.0) *(idx+1)
					# print (robotMarker.color.r, robotMarker.color.g)
					robotMarker.color.r = 0
					robotMarker.color.g = 1
					robotMarker.color.b = 0
					robotMarker.color.a = 1
					

					#print (self.markerArray.markers)
					#print ('**********************')
					robotMarker.lifetime = rospy.Duration(1)
					markerArray.markers.append(robotMarker)
					#print (self.markerArray.markers)
					
					print (CLASSES[cls_ind])
					print ('3D Translation:  ', all_trans[idx])
					print ('3D Rotation:    ', all_rot[idx])
				print('publish_markers')
				self.publisher.publish(markerArray.markers)
			else:
				print ('******* There is no obejct detected ********')
						
			Results_image_name = "/home/xiaoqiangyan/Documents/MyData/Detection/"+str(self.rgb_counter) + ".png"
			cv2.imwrite(Results_image_name, show_im)
			#rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			#Results_image_name = "/home/xiaoqiangyan/Documents/MyData/Detection/"+str(self.rgb_counter) + "_v.png"
			#cv2.imwrite(Results_image_name, rgb_image)

		except CvBridgeError, e:
			print (e)

	# def depth_image_callback(self,data):
	#     #http://answers.ros.org/question/58902/how-to-store-the-depth-data-from-kinectcameradepth_registreredimage_raw-as-gray-scale-image/
	#     #if self.have_depth_image == False:
	#     try:
	# 		filename = "/home/xiaoqiangyan/data/"
	# 		self.depth_counter += 1
	#
	# 		image_depth_name     = filename + '_depth_' +  str(self.depth_counter)+'.jpg'
	# 		image_np_matrix_name_txt = filename + '_depth_' +  str(self.depth_counter)+'.txt'
	# 		image_np_matrix_name = filename + '_depth_' +  str(self.depth_counter)
	#
	# 		depth_image_input = self.bridge.imgmsg_to_cv2(data, '16UC1') #32FC1 does not work with asus somehow.
	# 		depth_image2 = np.array(depth_image_input,dtype=np.int16)
	#
	# 		self.depth_image = depth_image2
	# 		#TODO choose one
	# 		#cv2.imwrite(image_depth_name, self.depth_image/MAX_DEPTH_RANGE*255) #this is used for RSS
	# 		#np.savetxt(image_np_matrix_name_txt, self.depth_image, fmt='%f')
	# 		np.save(image_np_matrix_name, self.depth_image) # I recommend this one
	#
	#     except CvBridgeError, e:
	#         print e
	#


def main():

	rospy.init_node('perception__parameters_node')
	ic = PerceptionParameterNode(rospy.get_name())
	r = rospy.Rate(ROS_RUN_RATE)
	try:
		while not rospy.is_shutdown():

			r.sleep()
	except KeyboardInterrupt:
		print "Shutting down"

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
