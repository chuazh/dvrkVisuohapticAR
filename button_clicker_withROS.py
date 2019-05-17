#!/usr/bin/env python

from Tkinter import *
import rospy
from std_msgs.msg import Bool
import copy
import dvrk
import PyKDL

rospy.init_node('advancerGUI')
pub = rospy.Publisher('advance_trial',Bool, queue_size = 1000)
rate = rospy.Rate(100)
master = Tk()
button_in = Bool()
global db_time 
db_time = 0
global curr_time
curr_time = 0
thresh = 0.1

p2 = dvrk.psm('PSM2')
m2 = dvrk.mtm('MTMR')
c = dvrk.console()

MTMR_pos = PyKDL.Frame()
PSM_pos = PyKDL.Frame()

def callback():
	global db_time 
	global curr_time
	db_time = copy.copy(curr_time)
	button_in.data = True
	
def reset_callback():

	#global MTMR_cart = PyKDL.Vector()
	#global MTMR_rot = PyKDL.Rotation()
	#global PSM_cart = PyKDL.Vector()
	#global PSM_rot = PyKDL.Rotation()


	c.teleop_stop()
	p2.move(PSM_pos)
	m2.move(MTMR_pos)
	c.telop_start()

def record_callback():

	MTMR_pos = m2.get_current_position()
	PSM_pos = p2.get_current_position()
	
	print(MTMR_pos)

f = Frame(master, height=60, width=60)
f.pack_propagate(0) # don't shrink
f.pack()
b = Button(master, text="OK", height = 30 , width = 30, command=callback)  
b.pack()
b2 = Button(master, text="Reset", height = 30 , width = 30, command=reset_callback) 
b2.pack()
b3 = Button(master, text="Reset", height = 30 , width = 30, command=record_callback) 
b3.pack()

while not rospy.is_shutdown():
	curr_time = rospy.get_time()
	master.update()
	diff = curr_time-db_time
	if diff > thresh:
		button_in.data = False
	
	pub.publish(button_in)
	rate.sleep()
	