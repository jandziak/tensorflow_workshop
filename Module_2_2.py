# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
#How to define and operate on matrices (2D tensors)
#diag, random_uniform, convert_to_tensor
#add, transpose, matmul,...

import tensorflow as tf
import numpy as np
sess = tf.Session()

#Declaration of matrices

#Identity
identity_matrix = tf.diag(np.ones(3))
print(sess.run(identity_matrix))

#Constant
const_matrix = tf.fill([2, 3], -1.0)
print(sess.run(const_matrix))

#Random Uniform
uniform_matrix = tf.random_uniform([3,2]) 
print(sess.run(uniform_matrix))
print(sess.run(uniform_matrix))

#Convert numpy to matrix
converted_matrix =  tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(converted_matrix))

#Add (substract) two martices (tensors)
print(sess.run(identity_matrix + identity_matrix))
print(sess.run(tf.add(identity_matrix, identity_matrix)))
print(sess.run(tf.subtract(identity_matrix, identity_matrix)))

#Matrix multiplication and transpose
uni_times_const = tf.matmul(uniform_matrix, const_matrix)
print(sess.run(tf.transpose(uni_times_const)))

#Inverse
two_identity = tf.add(identity_matrix, identity_matrix)
print(sess.run(tf.matrix_inverse(two_identity)))

#Determinant
#print(sess.run(tf.matrix_determinant(uni_times_const)))

# Eigenvalues and Eigenvectores
uniform_matrix_2 = tf.random_uniform([3,3])
print(sess.run(tf.self_adjoint_eig(uniform_matrix_2)))

### Exercise modelue_1_2

# Create constant_matrix with dim n_row = 3, ncol = 2 
# Create rn_matrix with random_normal values with dim nrow_2, n_col = 3
# Create diag_matrix dim 3x3 with 4 on the diagonal
# Multiply constant matrix and rn_matrix
# Add diag_matrix
# Print results