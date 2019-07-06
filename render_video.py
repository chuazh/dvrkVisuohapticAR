#!/usr/bin/env python

import rospy
import cv2,cv_bridge
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Wrench, Pose
import argparse
import numpy as np
import copy
import pyaudio

class camera_module:
	def __init__(self,side,condition,debug,palpate):
		self.bridge = cv_bridge.CvBridge()
		self.side = side
		self.text = "Initialized"
		
		# Declare our subscribers
		sub = rospy.Subscriber('/force_msg',String,self.msg_callback, queue_size = 100)
		sub_force = rospy.Subscriber('/force_sensor',Wrench,self.force_callback, queue_size = 100)
		sub_cam_reset = rospy.Subscriber('/cam_reset',Bool,self.cam_reset_callback, queue_size = 100)
		sub_teleop_pedal = rospy.Subscriber('/dvrk/footpedals/coag',Joy,self.teleop_callback, queue_size = 100)
		sub_PSM_pos = rospy.Subscriber('/ep_pose',Pose,self.get_position, queue_size = 100)
		#sub_advance_trial = rospy.Subscriber('/advance_trial',Bool,self.advance_callback2,queue_size = 100)
		sub_advance_trial = rospy.Subscriber('/dvrk/footpedals/camera',Joy,self.advance_callback,queue_size = 100)
		sub_catch = rospy.Subscriber('/catch_trial',Bool,self.catch_callback,queue_size =100)
		
		if self.side == "left":
			self.image_sub = rospy.Subscriber('/camera/left/image_color',Image,self.image_callback)
		else:
			self.image_sub = rospy.Subscriber('/camera/right/image_color',Image,self.image_callback)
			
			
		self.font                   = cv2.FONT_HERSHEY_SIMPLEX
		self.fontScale              = 0.75
		self.fontColor              = (255,255,255)
		self.lineType               = 2
		self.debug = debug
		self.palpate = palpate
		self.condition = condition
		self.displacement = 0
		self.force = 0
		
		self.current_time = 0
		self.trial_begin_time = 0
		self.time_old = 0
		
		self.advance_flag = False
		self.trial_begin = False
		self.catch_flag = False
		
		if self.side == 'right':
			self.sound_flag = 0
			self.sound = pyaudio.PyAudio()
			self.framerate = 44100
			# open stream using callback
			self.stream = self.sound.open(format = self.sound.get_format_from_width(1),
		        channels = 1,
		        rate= int(self.framerate),
		        frames_per_buffer = 16000,
		        output=True,
		        stream_callback = self.audio_callback)
		
		'''
		#stuff for color masking
		self.color_timer_overlay = cv2.imread("overlay.png", cv2.IMREAD_UNCHANGED);
		self.hsv_im = cv2.cvtColor(self.color_timer_overlay,cv2.COLOR_BGR2HSV) # get a HSV version of the image
		gray_im = cv2.cvtColor(self.color_timer_overlay,cv2.COLOR_BGR2GRAY) #get a grayscale version for thresholding
		# generate thesholding mask
		ret,mask = cv2.threshold(gray_im,127,255,cv2.THRESH_BINARY)
		self.thresh_idx = np.where(mask == 255)
		# grab alpha mask from the PNG image
		m,n = np.shape(self.color_timer_overlay[:,:,3])
		self.alpha_mask = np.reshape(1-(self.color_timer_overlay[:,:,3].astype(float)/255),(m,n,1))
		self.alpha_mask_inv = np.reshape(self.color_timer_overlay[:,:,3].astype(float)/255,(m,n,1))
		'''
		
	def image_callback(self,msg):
		
		self.current_time = rospy.get_time()
		frame_rate = 1/(self.current_time-self.time_old)
		self.time_old = self.current_time
		
		image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8") #transfer image from ros stream into open cv
		
		if self.palpate:
			image = self.crop_and_move(image,0,720)
		else:
			image = self.crop_and_move(image,self.displacement,720) # crop the image and perturb it randomly
		
		aug_image = self.augment_image(image) # add our augmented reality effects
		
		#aug_image = self.decrease_brightness(aug_image,1)
			
		if self.side == "left":
			cv2.namedWindow("left", flags= 16)
			cv2.imshow("left",aug_image)
			cv2.moveWindow("left", 2000, 0)
			cv2.setWindowProperty("left", cv2.WND_PROP_FULLSCREEN, 1) 
				
		elif self.side == "center":
			
			cv2.namedWindow("center", flags= 16)
			cv2.imshow("center",aug_image)
			
		elif self.side == "center2":
			
			cv2.namedWindow("center2", flags= 16)
			cv2.imshow("center2",aug_image)
		
		else:
			cv2.namedWindow("right", flags= 16)
			cv2.imshow("right",aug_image)
			cv2.moveWindow("right", 3000, 0)
			#cv2.setWindowProperty("right", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
			cv2.setWindowProperty("right", cv2.WND_PROP_FULLSCREEN, 1)
		
		cv2.waitKey(3)
		
	def augment_image(self,image):
	
		time = self.current_time-self.trial_begin_time
		
		height,width,channel = image.shape
		
		aug_height = (720-height)/2
		
		bgr_upper = np.zeros((1,3))
		bgr_lower = np.zeros((1,3))
		
		''' Stuff for borders...quite ugly tbh
		# read in the average value of the top row of the image
		for i in range(0,width):
			bgr_upper = bgr_upper + image[0,i]
			bgr_lower = bgr_lower + image[-1,i]
		
		bgr_mean_upper = bgr_upper/(width) 
		bgr_mean_lower = bgr_lower/(width)
		
		#outputImage = cv2.copyMakeBorder(image, aug_height, 0 , 0, 0, cv2.BORDER_REPLICATE )
		#outputImage = cv2.copyMakeBorder(image, aug_height, 0 , 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])#bgr_mean_upper[0]
		#outputImage = cv2.copyMakeBorder(outputImage, 0, aug_height, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0]) #bgr_mean_lower[0]
		'''
		
		if self.condition == "nohaptics":
			#outputImage = self.draw_force_bar_color(image)
			outputImage = self.threshold_and_color(image)
			#if time < 5:
				#print(time)
		else:
			outputImage = image
		
		if self.side == 'right':
			if time > 5 and self.trial_begin == True:
				self.stream.start_stream()
			elif self.trial_begin == False:
				self.trial_begin_time = copy.copy(self.current_time)
				self.stream.stop_stream()
		
		# get boundary of this text
		textsize = cv2.getTextSize(self.text, self.font, self.fontScale, self.lineType)[0]

		# get coords based on boundary
		textX = (outputImage.shape[1] - textsize[0]) / 2
		#textX = 10
		textY = 50
	
		bottomLeftCornerOfText = (textX,textY)	
		bottomLeftCornerOfText2 = (textX,textY+20)	
		
		if self.advance_flag == True:
			#self.text = " "	
			self.advance_flag = False
			self.trial_begin = False
		
		if self.debug == True:
			self.text = '%.3f' % self.force + ',' + '%.3f' % self.forceY + ',' + '%.3f' % self.forceZ
			self.text2 = '%.3f' % self.x + ',' + '%.3f' % self.y+ ',' + '%.3f' % self.z
			#self.text2 = ' '
			cv2.putText(outputImage,self.text2, bottomLeftCornerOfText2, self.font, self.fontScale,self.fontColor,self.lineType)
			
		cv2.putText(outputImage,self.text, bottomLeftCornerOfText, self.font, self.fontScale,self.fontColor,self.lineType)
		
		return outputImage
		
	def msg_callback(self,data):
		
		self.text = str(data)
		self.text = self.text.replace('data: ', '')
		
		# use this if statement to check if we are starting our trial
		if ('Go!!!' in self.text and self.trial_begin==False):
    			self.trial_begin_time = copy.copy(self.current_time)
    			self.trial_begin = True
		
	def teleop_callback(self,data):
		if self.trial_begin:
			self.text = " "		# resets the screen..i.e clears it. This has to be disabled if we want to display time.	
			
	def advance_callback(self,data):
		if data.buttons[0] > 0.5:
			self.advance_flag = True
	
	def advance_callback2(self,data):
		if data.data == True:
			self.advance_flag = True
	
	def catch_callback(self,data):
		self.catch_flag = data.data
		
		
	def audio_callback(self,in_data,frame_count,time_info,status):
		
		time = self.current_time-self.trial_begin_time #get the current timing
		numTimeSteps = 1.0/self.framerate * frame_count
		t = np.linspace(time, time+numTimeSteps,frame_count,endpoint=False)
		frequency = 4000.0
		omega = 2.0*np.pi*frequency
		data = np.sin(t*omega)
		data = np.array((data+1.0)*127.5,dtype=np.int8).tostring()

		cont_flag = pyaudio.paContinue
		
		return (data,cont_flag)
				 
	def cam_reset_callback(self,data):
		self.displacement = np.random.random_integers(-30,100,1)[0]
		if self.trial_begin:
			self.trial_begin = False # if the user didn't advance the trial by flipping the advance_trial flag and the timer expired
	
	def get_position(self,data):
		self.x = data.position.x
		self.y = data.position.y
		self.z = data.position.z
	
	def decrease_brightness(self,img, value):
	
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv)

		lim = 0 + value
		v[v < lim] = 0
		v[v >= lim] -= value

		final_hsv = cv2.merge((h, s, v))
		img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
		return img
	    
	def crop_and_move(self, image, displacement, crop_size):
	
		#our image width by default is 960px. We crop to 720px. This leaves 110px on either side that we can exploit. 
		
		height,width,channel = image.shape
		
		mid_pix = width/2
		
		# check max displacement and threshold
		
		# get the image center
		image_center = mid_pix + displacement
		
		# crop the image
		cropped_image = image[:,image_center-crop_size/2:image_center+crop_size/2]
		
		'''
		print("image center: " + str(image_center))
		print("image left: " + str(image_center-crop_size/2))
		print("image right: " + str(image_center+crop_size/2))
		'''
		return cropped_image
		
	def draw_force_bar(self,image):
		
		im_height,im_width,channel = image.shape
		
		rectangle_base = int(0.9*im_height)
		
		width = 20
		
		cal_height = 500
		cal_force = 10
		height = int(self.force*cal_height/cal_force)
		
		cv2.rectangle(image, (10,rectangle_base), (10+width,rectangle_base-height), (0,255,0),-1,4)
		
		return image
	
	def draw_force_bar_color(self,image):
		
		im_height,im_width,channel = image.shape
		
		rectangle_base = int(0.9*im_height)
		height = 50
		width = 20
		scaled_force = self.force*((255/2)/3)
		scaled_force = np.clip(scaled_force,0,255/2)
		cv2.rectangle(image, (10,rectangle_base), (10+width,rectangle_base+height), (255/2+scaled_force,0,255/2-scaled_force),-1,4)
		
		return image
		
	def force_callback(self,data):
		
		self.force = data.force.z
		self.forceY =  -data.force.y
		self.forceZ = -data.force.x
	
	def time_varying_color_border(self,image,t):
		
		trial_time = 3
		color_change_divisor = 8 # to get direct green to red the divisor should be equal to trial time
		
		h_g = np.uint8(60) # hue value for green
		h_r = np.uint8(0) # hue value for red
		val = 255 
		
		if self.trial_begin == False:
			hue = h_g
			val = np.uint(0)
		elif t < trial_time:
			hue = np.uint8(h_g-h_g*t/color_change_divisor)
			val = np.uint8(255)

		else:
			hue = h_r
		
		self.hsv_im[self.thresh_idx[0],self.thresh_idx[1],0:3] = np.array([hue,255,val],dtype=np.uint8)
		bgr_im = cv2.cvtColor(self.hsv_im,cv2.COLOR_HSV2BGR)
		
		cv2.addWeighted(bgr_im,0.3,image,1-0.3,0,image)
		
		#background = self.alpha_mask*image[:,:,0:3]
		#foreground = self.alpha_mask_inv*bgr_im
		#im_out = (foreground+background)/255
		
		return image
		
	def threshold_and_color(self,image):
	
		#grayscaled = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert to grayscale
		#retval,threshold = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY) # threshold the image
		#threshold = cv2.adaptiveThreshold(grayscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
		
		if self.palpate: 
			lower = (115,75,150)
		else:
			lower = (115,75,85)
		
		upper = (191,255,255)
		
		image_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV) #convert our image to HSV
		mask = cv2.inRange(image_hsv,lower,upper)
		idx = np.where(mask == 255)
		
		''' code to make the blacks really black'''
		'''
		black_lower = (0,0,0)
		black_upper = (255,255,40)
		black_mask = cv2.inRange(image_hsv,black_lower,black_upper)
		black_idx = np.where(black_mask == 255)

		image[black_idx] = np.array([0,0,0])
		'''
		
		#idx = np.where(threshold == 255)
		
		#scaled color according to force
		#calib_constant = 1.5/40.0
		init = 255
		if self.catch_flag == False:
			sat_force = 9.0
		else:
			sat_force = 200.0
		calib_constant = sat_force/init
		
		
		if self.palpate == 1:
			s = int(init+self.force/calib_constant) # this is for in compression
		else:
			s = int(init-self.force/calib_constant) # this is for in tension
		
		s = np.clip(s, 0, 255)
		#print(np.linalg.norm(np.array([self.force,self.forceY,self.forceZ])))
		
		if np.abs(self.force) < sat_force:
			image_hsv[idx] = np.array([128,s,255]) # color the pixels with our desired value
		else:
			image_hsv[idx] = np.array([200,255,50]) # color the pixels with our desired value
		
				
		overlay = cv2.cvtColor(image_hsv,cv2.COLOR_HSV2RGB) # convert this image back to BGR and use it as an overlay
		alpha = 0.3 # 0.5
		cv2.addWeighted(overlay, alpha, image, 1 - alpha,0, image)
		
		return image
	
	def color_timer(image,time_diff):
		self.color_timer_overlay 
		

parser = argparse.ArgumentParser()
parser.add_argument("side", help="specify display side",type=str)
parser.add_argument("haptics", help="condition",type=str)
parser.add_argument("--palpate", help="palpate condition", type=int)
parser.add_argument("--debug", type=bool)
args = parser.parse_args()
nodename = "cameraDisplay_" + args.side
rospy.init_node(nodename)
rate = rospy.Rate(1000)
np.random.seed(9001)
if args.debug: 
	cam = camera_module(args.side,args.haptics,True,args.palpate)
else:
	
	cam = camera_module(args.side,args.haptics,False,args.palpate)
rospy.spin()
		



		