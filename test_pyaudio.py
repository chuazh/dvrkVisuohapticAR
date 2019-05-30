#!/usr/bin/env python
import pyaudio
import numpy as np
import time
import copy

def audio_callback(in_data,frame_count,time_info,status):	
	#time = self.current_time-self.trial_begin_time #get the current timing
	deltaT = 1.0/framerate * (frame_count)
	t = np.linspace(time_elapsed, time_elapsed+deltaT,frame_count,endpoint=False)
	frequency = 300*np.sin(t)+ 4000 #5000.0
	omega = 2.0*np.pi*frequency
	data = np.sin(t*omega)
	data = np.array((data+1.0)*127.5,dtype=np.int8).tostring()
	'''
	if sound_flag == 0:
		cont_flag = pyaudio.paComplete
	else:
		cont_flag = pyaudio.paContinue'''
	cont_flag = pyaudio.paContinue

	return (data,cont_flag)


time_init = time.time()
time_old = 0

sound_flag = 0
sound = pyaudio.PyAudio()
framerate = 44100
# open stream using callback
stream = sound.open(format = sound.get_format_from_width(1),frames_per_buffer=16000,channels = 1,rate= int(framerate),output=True,stream_callback = audio_callback)
#stream.start_stream()

active = True

while active:

	time_elapsed = time.time()-time_init
	stream.start_stream()
	'''
	if time_elapsed > 2 and time_elapsed < 4 and sound_flag == 1:
		stream.stop_stream()
		print(time_elapsed)
		sound_flag = 0
	if time_elapsed > 4 and sound_flag == 0:
		print(time_elapsed)
		sound_flag = 1
		stream.start_stream()
	if time_elapsed >6:
		stream.stop_stream()
		print(time_elapsed)
		sound_flag = 0
		active = False
	'''

stream.close()
sound.terminate()
print('done')