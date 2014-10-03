###   https://github.com/zbanbosuiyue/Python_learning/  ###

import csv
import math
import random
import string

####										####
#### 	Prepare data for Neuron Network 					####
####										####
# Load CSV File
def fitem(item):
	item = item.strip()
	try:
		item = float(item)
	except ValueError:
		pass
	return item

def readcsv(filename):
	with open(filename, 'r') as csvin:
		reader=csv.DictReader(csvin)
		data={k.strip():[fitem(v)] for k,v in reader.next().items()}
		for line in reader:
			for k,v in line.items():
				k = k.strip()
				data[k].append(fitem(v))
	return data

# Read crabs.csv
data=readcsv('crabs.csv')
# Define 5 Inputs FL,RW,CL,CW,BD
FL = data['FL']		#FL==frontal lobe size (mm).
RW = data['RW']		#RW==rear width (mm).
CL = data['CL']		#CL==carapace length (mm).
CW = data['CW']		#CW==carapace width (mm).
BD = data['BD']		#BD==body depth (mm).

# Define supervise result, change sex type to '1' and '0'
i=0
targets = data['sex']
for index,item in enumerate(targets):
	if item == 'M':
		targets[index] = 0.5
	else:
		targets[index] = -0.5

new_targets = []		
for i in range(len(targets)):
	new_targets.append([])
	new_targets[i].append(targets[i])
	
# Organize data
inputs = []
for i in range(len(FL)):
	inputs.append([])
	#inputs[i].append(FL[i])
	inputs[i].append(RW[i])
	#inputs[i].append(CL[i])
	inputs[i].append(CW[i])
	#inputs[i].append(BD[i])
	
new_data = []
for i in range(len(new_targets)-1):
	new_data.append([])
	new_data[i].append(inputs[i])
	new_data[i].append(new_targets[i])

# Stochastic Data
ran_data=[]
for i in range(len(new_data)):
	ran_data.append(new_data[random.randint(0,150)])

# Random Number
def rand(a, b):
	return (b - a) * random.random() + a

# Sigmoid or Tanh function
def sigmoid(x):
	return math.tanh(x)

# Derivative of Sigmoid
def d_sigmoid(x):
	return 1.0 - x**2



####							####	
####		INITIALIZATION				####
####							####


###Initialize Neuron NetWorks###
class NeuronNetwork:
	def __init__(self):
		# Initialize each weights and set them to random value
		self.weight=[]
		self.weight.append(rand(-0.1,0.1))
		self.weight.append(rand(-0.1,0.1))
			
	#### Calculate the output	####
	def feedForward(self, inputs):
		self.input=inputs
		sum=0
		for i in range(len(self.weight)):
			sum = sum + self.input[i]*self.weight[i]
		self.out = sigmoid(sum)
		return self.out
		

	#### Back Propagation Progress 								 ####
	def backPropagation(self, targets, N):
		# Calculate the error
		error = targets[0]-self.out
		
		# Calculate the delta
		delta = d_sigmoid(self.out) * error

		# Update weights #
		for i in range(len(self.weight)):
			self.weight[i] += N*delta*self.input[i]
		
		# Return system error
		sys_error=0.5*(error)**2
		return sys_error

	# Train
	def train(self, input,iterations=5000, N=0.00001):
		for i in range(iterations):
			error = 0.0
			for j in input:
				inputs = j[0]
				targets = j[1]
				self.target= targets
				self.feedForward(inputs)
				#print self.weight
				#print inputs
				error = error + self.backPropagation(targets, N)
			if i % 100 ==0:	
				print ('error= %f' %error)
				
			if error<=9:
				break
		print ('weight=',self.weight)

	# Test
	def test(self, data):
		correct_n=0.0
		wrong_n=0.0
		for i in data:
		
			test_target=''
			if self.feedForward(i[0])>0:
				test_target='Male'
			else:
				test_target='Female'
			if i[1][0]==0.5:
				sex='Male'
			else:
				sex='Female'
			print(i[0], '->', test_target,'target-->',sex)
			if test_target==sex:
				correct_n+=1
			else:
				wrong_n+=1
		accuracy='{percent:.2%}'.format(percent=correct_n/(correct_n+wrong_n))
		print('Correct= %i. Wrong= %i' %(correct_n,wrong_n))
		print('Accuracy= %s' %(accuracy))
		
def run():
	n = NeuronNetwork()
	n.train(ran_data[0:200])
	n.test(new_data[101:200])

if __name__ == '__main__':
	run()
