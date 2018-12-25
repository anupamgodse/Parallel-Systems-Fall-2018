#!/bin/python

#Single Author info:
#vphadke Vandan Vasant Phadke

#Group info:

#angodse Anupam Godse
#yjkamdar Yash J Kamdar

#Import libraries for simulation
import tensorflow as tf
import numpy as np
import sys
import time
import horovod.tensorflow as hvd

#Imports for visualization
import PIL.Image

def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Outputs an array as a dat file."""
  # We dont have to print the last 3 rows as they are placeholders
  n = len(a) - 3
  with open("lake_c_" + str(hvd.rank()) + ".dat","w") as f:
      for i in range(n):
          for j in range(n):
              f.write(str((i * 1.0)/n) + " " + str((j*1.0)/n) + " " + str(a[i][j]) + "\n")

hvd.init()
sess = tf.InteractiveSession()

# Computational Convenience Functions
def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
#  5 point stencil #
  five_point = [[0.0, 1.0, 0.0],
                [1.0, -4., 1.0],
                [0.0, 1.0, 0.0]]

#  9 point stencil #
  nine_point = [[0.25, 1.0, 0.25],
                [1.00, -5., 1.00],
                [0.25, 1.0, 0.25]]

#  13 point stencil #
  thirteen_point = [[0,0,0,0.125,0,0,0],
                [0,0,0,0.25,0,0,0],
                [0,0,0,1.0,0,0,0],
                [0.125,0.25,1.0,-5.5,1.0,0.25,0.125],
                [0,0,0,1.0,0,0,0],
                [0,0,0,0.25,0,0,0],
                [0,0,0,0.125,0,0,0]]

  laplace_k = make_kernel(thirteen_point)
  return simple_conv(x, laplace_k)

# Define the PDE
if len(sys.argv) != 4:
	print "Usage:", sys.argv[0], "N npebs num_iter"
	sys.exit()

N = int(sys.argv[1])
npebs = int(sys.argv[2])
num_iter = int(sys.argv[3])

# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init  = np.zeros([N+3, N], dtype=np.float32)
ut_init = np.zeros([N+3, N], dtype=np.float32)

# Some rain drops hit a pond at random points
for n in range(npebs):
  # For rank 0, we dont have to generate pebbles in the last three rows
  if hvd.rank() == 0:
    a,b = np.random.randint(0, N, 2)
    u_init[a,b] = np.random.uniform()
  # For rank 1, we dont have to generate pebbles in the first three rows
  else:
    a = np.random.randint(3, N+3, 1)
    b = np.random.randint(0, N, 1)
    u_init[a,b] = np.random.uniform()

# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Discretized PDE update rules
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

#create send and receive buffers
send_buf  = np.zeros([3, N], dtype=np.float32)
recv1_buf  = np.zeros([3, N], dtype=np.float32)
recv0_buf  = np.zeros([3, N], dtype=np.float32)

Send_Buffer  = tf.Variable(send_buf,  name='Send_Buffer')
Recv0_Buffer  = tf.Variable(recv0_buf,  name='Recv0_Buffer')
Recv1_Buffer  = tf.Variable(recv1_buf,  name='Recv1_Buffer')

# Operation to update the state
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))

# Fill the last (or first) 3 rows value in sendBuffer
lake_update1 = None
if hvd.rank() == 0:
  lake_update1 = tf.scatter_update(Send_Buffer, [0,1,2], Ut[N-3:N, :])
else:
  lake_update1 = tf.scatter_update(Send_Buffer, [0,1,2], Ut[3:6, :])

# Send to the other rank
bcast = tf.group(
  tf.assign(Recv1_Buffer, hvd.broadcast(Send_Buffer, 0)),  #Rank 0's send_buffer to Rank 1's recv
  tf.assign(Recv0_Buffer, hvd.broadcast(Send_Buffer, 1)))  #Rank 1's send_buffer to Rank 0's recv

# Update the current lake state based on the received data
lake_update2 = None
if hvd.rank() == 0:
  lake_update2 = tf.scatter_update(Ut, [N,N+1,N+2], Recv0_Buffer[:, :])
else:
  lake_update2 = tf.scatter_update(Ut, [0,1,2], Recv1_Buffer[:, :])

# Initialize state to initial conditions
tf.global_variables_initializer().run()

# Run num_iter steps of PDE
start = time.time()
for i in range(num_iter):
  # Step simulation
  step.run({eps: 0.06, damping: 0.03})
  # Update send buffer
  sess.run(lake_update1)
  # Broadcast variables
  bcast.run()
  # Update received variables
  sess.run(lake_update2)

end = time.time()
print('Elapsed time: {} seconds'.format(end - start))
DisplayArray(U.eval(), rng=[-0.1, 0.1])
