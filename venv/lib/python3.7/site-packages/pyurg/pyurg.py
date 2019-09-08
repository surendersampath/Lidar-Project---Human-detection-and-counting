#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division
import serial
import re
import math
import time


class UrgDevice(serial.Serial):
	def __init__(self):
		super(serial.Serial, self).__init__()

	def __del__(self):
		self.laser_off()
		self.close()

	def connect(self, port, baudrate=115200, timeout=0.1):
		'''
		Connect to URG device
		port	  : Port or device name. ex:/dev/ttyACM0, COM1, etc...
		baudrate  : Set baudrate. ex: 9600, 38400, etc...
		timeout	  : Set timeout[sec]
		'''
		self.port = port
		self.baudrate = baudrate
		self.timeout = timeout
		try:
			self.open()
		except:
			print('Could not open', port, 'at', baudrate)
			return False

		self.set_scip2()

		ret = False
		while not ret:
			ret = self.get_parameter()
			print('get_parameter()')

		if not self.laser_on():
			print('Could not turn on laser')
			return False

		print(self)

		return True

	def printInfo(self):
		if self.pp_params is None:
			print('Please connect() first')
			return

		print('========================================')
		print('Hokuyo URG-o4LX')
		print('----------------------------------------')
		print('Port: {} @ {}'.format(self.port, self.baudrate))
		print('Protocol: SCIP2.0')
		print('Version:', self.get_version())
		print('Params:', self.pp_params)
		print('Scan Time [sec]:', self.scan_sec())
		print('')

	def flush_input_buf(self):
		'''Clear input buffer.'''
		self.flushInput()

	def send_command(self, cmd):
		'''Send command to device.'''
		self.write(cmd)

	def __receive_data(self):
		return self.readlines()

	def set_scip2(self):
		'''Set SCIP2.0 protcol'''
		self.flush_input_buf()
		self.send_command('SCIP2.0\n')
		return self.__receive_data()

	def get_version(self):
		'''Get version information.'''
		if not self.isOpen():
			return False

		self.flush_input_buf()
		self.send_command('VV\n')
		get = self.__receive_data()
		return get

	def get_parameter(self):
		'''
		Get device parameter and set self.pp_params
		return: True/False
		'''
		ret = self.isOpen()
		if not ret:
			return False

		self.send_command('PP\n')
		time.sleep(0.1)

		get = self.__receive_data()

		# check expected value
		if not (get[:2] == ['PP\n', '00P\n']):
			return False

		# pick received data out of parameters
		self.pp_params = {}
		for item in get[2:10]:
			tmp = re.split(r':|;', item)[:2]
			self.pp_params[tmp[0]] = tmp[1]
		return True

	def laser_on(self):
		'''Turn on the laser.'''
		if not self.isOpen():
			return False

		self.send_command('BM\n')

		get = self.__receive_data()

		if not(get == ['BM\n', '00P\n', '\n']) and not(get == ['BM\n', '02R\n', '\n']):
			return False
		return True

	def laser_off(self):
		'''Turn off the laser.'''
		if not self.isOpen():
			return False

		self.flush_input_buf()
		self.send_command('QT\n')
		get = self.__receive_data()

		if not(get == ['QT\n', '00P\n', '\n']):
			return False
		return True

	def __decode(self, encode_str):
		'''Return a numeric which converted encoded string from numeric'''
		decode = 0

		for c in encode_str:
			decode <<= 6
			decode &= ~0x3f
			decode |= ord(c) - 0x30

		return decode

	def __decode_length(self, encode_str, byte):
		'''Return leght data as list'''
		data = []

		for i in range(0, len(encode_str), byte):
			split_str = encode_str[i:i+byte]
			data.append(self.__decode(split_str))

		return data

	def index2rad(self, index):
		'''Convert index to radian and reurun.'''
		rad = (2.0 * math.pi) * (index - int(self.pp_params['AFRT'])) / int(self.pp_params['ARES'])
		return rad

# 	def create_capture_command(self):
# 		'''create capture command.'''
# 		cmd = 'GD' + self.pp_params['AMIN'].zfill(4) + self.pp_params['AMAX'].zfill(4) + '01\n'
# 		return cmd

	def scan_sec(self):
		'''Return time of a cycle.'''
		rpm = float(self.pp_params['SCAN'])
		return (60.0 / rpm)

	def capture(self):
		# Receive lenght data
		# cmd = self.create_capture_command()
		cmd = 'GD' + self.pp_params['AMIN'].zfill(4) + self.pp_params['AMAX'].zfill(4) + '01\n'
		self.flush_input_buf()
		self.send_command(cmd)
		time.sleep(0.1)
		get = self.__receive_data()

		# checking the answer
		if not (get[:2] == [cmd, '00P\n']):
			return [], -1

		# decode the timestamp
		tm_str = get[2][:-1]  # timestamp
		timestamp = self.__decode(tm_str)

		# decode length data
		length_byte = 0
		line_decode_str = ''
		if cmd[:2] == ('GS' or 'MS'):
			length_byte = 2
		elif cmd[:2] == ('GD' or 'MD'):
			length_byte = 3
		# Combine different lines which mean length data
		NUM_OF_CHECKSUM = -2
		for line in get[3:]:
			line_decode_str += line[:NUM_OF_CHECKSUM]

		# Set dummy data by begin index.
		self.length_data = [-1 for i in range(int(self.pp_params['AMIN']))]
		self.length_data += self.__decode_length(line_decode_str, length_byte)
		return (self.length_data, timestamp)


def main():
	urg = UrgDevice()
	if not urg.connect():
		print 'Connect error'
		exit()

	for i in range(10):
		data, tm = urg.capture()
		if data == 0:
			continue
		print len(data), tm


# if __name__ == '__main__':
# 	main()
