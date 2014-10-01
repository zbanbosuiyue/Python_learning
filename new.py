import csv
import math
import random
import string
from sys import argv

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
	
def read_weight(filename):
	with open(filename,'r') as csvin:
		reader=csv.reader(csvin, delimiter=',')
		return list(reader)
		
# If old_work=1, read weight.csv
# If w_sign=1, write to weight.csv	
old_work=0
w_sign=0

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
		targets[index] = 1
	else:
		targets[index] = 0

new_targets = []		
for i in range(len(targets)):
	new_targets.append([])
	new_targets[i].append(targets[i])
	
# Organize data
inputs = []
for i in range(len(FL)):
	inputs.append([])
	inputs[i].append(FL[i])
	#inputs[i].append(RW[i])
	#inputs[i].append(CL[i])
	inputs[i].append(CW[i])
	#inputs[i].append(BD[i])
	
new_data = []

for i in range(len(new_targets)-1):
	new_data.append([])
	new_data[i].append(inputs[i])
	new_data[i].append(new_targets[i])
	
ran_data=[]
	for i in range(len(new_data)):
		ran_data.append(new_data[random.randint(0,150)])

# Stochastic Data
ran_data=[]
for i in range(len(new_data)):
	ran_data.append(new_data[random.randint(0,100)])

# Random Number
def rand(a, b):
	return (b - a) * random.random() + a

# Sigmoid or Tanh function
def sigmoid(x):
	return math.tanh(x)

# Derivative of Sigmoid
def d_sigmoid(x):
	return 1.0 - x**2

# Define make matrix function
def makeMatrix(row,col,value=0):
	m=[]
	for i in range(row):
		m.append([value] * col)
	return m


####							####	
####		INITIALIZATION				####
####							####


###Initialize Neuron NetWorks###
class NeuronNetwork:
	def __init__(self, num_in, num_h, num_out):
		# Set numbers of nodes for input, hidden 1, hidden 2 and output layers.
		# Each layer would have bias execpt output layer.
		self.num_in = num_in+1
		self.num_h= num_h+1
		self.num_out = num_out

		# Initialize these nodes, and set their intial output after activation equal to 1
		self.out_in = [1.0]*self.num_in
		self.out_h = [1.0]*self.num_h
		self.out_out = [1.0]*self.num_out

		# Initialize each weights and set them to random value
		self.w_in_h = makeMatrix(self.num_in, self.num_h)
		self.w_h_out = makeMatrix(self.num_h, self.num_out)
		
		
		# If old_work==1, read weight from file. Else random weight.
		for i in range(self.num_in):
			for j in range(self.num_h):
				if old_work==1:
					data=read_weight('weight.csv')
					self.w_in_h[i][j]=float(data[i][j])
				else:
					self.w_in_h[i][j] = rand(-1,1)

		for i in range(self.num_h):
			for j in range(self.num_out):
				if old_work==1:
					data=read_weight('weight.csv')
					self.w_h_out[i][j]=float(data[i+3][j])
				else:
					self.w_h_out[i][j] = rand(-1,1)
		
		print self.w_h_out
		
		
	#### Calculate output of each nodes 			####
	#### Sequence is like input-->hidden-->output 	####
	def update(self, input):
		### Number of inputs should be equal to nodes of input layer
		if len(input) != self.num_in - 1:
			raise ValueError('Number of inputs should be equal to nodes of input layer')

		# Each node outputs of input layer == system inputs
		for i in range(self.num_in - 1):
			self.out_in[i] = input[i]

		# Each node outputs of hidden layer == sigmoid(Sum of input layers output with weights)
		for j in range(self.num_h):
			sum = 0.0
			for i in range(self.num_in):
				sum += self.out_in[i] * self.w_in_h[i][j]
			self.out_h[j] = sigmoid(sum)

		# Each node outputs of output layer == sigmoid(Sum of hidden layers outputs with weights)
		for j in range(self.num_out):
			sum = 0.0
			for i in range(self.num_h):
				sum += self.out_h[i] * self.w_h_out[i][j]
			self.out_out[j] = sigmoid(sum)
		
		# Return the system output	
		return self.out_out[:]

	#### Back Propagation Progress 								 ####
	#### Calculate daltas, and sequence: output-->hidden-->input ####
	def backPropagation(self, targets, N):
		# Num of targets should be equal to nodes of output layer
		if len(targets) != self.num_out:
			raise ValueError('Wrong number of target values')

		# Calculate the delta of each nodes of output layer
		deltas_out = [0.0] * self.num_out
		for i in range(self.num_out):
			error = targets[i] - self.out_out[i]
			deltas_out[i] = d_sigmoid(self.out_out[i])* error

		# Calculate the delta of each nodes of hidden layer
		deltas_h = [0.0] * self.num_h
		for i in range(self.num_h):
			error = 0.0
			for j in range(self.num_out):
				error += deltas_out[j] * self.w_h_out[i][j]
			deltas_h[i] = d_sigmoid(self.out_h[i]) * error
		
		### Update weights ####
		# Update w_h_out
		for i in range(self.num_h):
			for j in range(self.num_out):
				self.w_h_out[i][j] += N*self.out_h[i]*deltas_out[j]

		# Update w_in_h
		for i in range(self.num_in):
			for j in range(self.num_h):
				self.w_in_h[i][j] += N*self.out_in[i]*deltas_h[j]

		# Return system error
		error = 0.0
		for i in range(len(targets)):
			error = error + 0.5*(targets[i]-self.out_out[i])**2
		return error

	####   Train  		####
	#### N is learning rate ####
	def train(self, input,iterations=100000, N=0.0001):
		for i in range(iterations):
			error = 0.0
			for j in input:
				inputs = j[0]
				targets = j[1]
				self.update(inputs)
				error = error + self.backPropagation(targets, N)
				
			if i % 100 == 0:
				print ('error= %f' %error)
				
			# if error small than a threshold just stop and save weight to weight.csv file	
			if error<=24:
				w_sign=1
				break
				
		# Write to weight.csv, wait for next time run.
		if w_sign==1:
			m=0
			target=open('weight.csv','wb')
			wr=csv.writer(target,dialect='excel')
			for i in self.w_in_h:
				wr.writerow(i)
				
			for i in self.w_h_out:
				wr.writerow(i)
			print self.w_h_out
		
		
	# Test
	def test(self, data):
		for i in data:
			print(i[0], '->', self.update(i[0]),'target-->',i[1])
		
def run():
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
	n = NeuronNetwork(2, 5, 1)
	n.train(ran_data[0:199])
	 
	#n.test(ran_data[101:199])

if __name__ == '__main__':
	run()
