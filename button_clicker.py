#!/usr/bin/env python

from Tkinter import *
import rospy
from std_msgs.msg import Bool
import copy

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


def callback():
	global db_time 
	global curr_time
	db_time = copy.copy(curr_time)
	button_in.data = True
	

f = Frame(master, height=60, width=60)
f.pack_propagate(0) # don't shrink
f.pack()
b = Button(master, text="OK", height = 30 , width = 30, command=callback)  
b.pack()

while not rospy.is_shutdown():
	curr_time = rospy.get_time()
	master.update()
	diff = curr_time-db_time
	if diff > thresh:
		button_in.data = False
	
	pub.publish(button_in)
	rate.sleep()
	