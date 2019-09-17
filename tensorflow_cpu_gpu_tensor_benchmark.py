import tensorflow as tf
import timeit, functools

def test(a,b):
	tf_matrix_multiplication_prod = tf.matmul(a, b)
	return sess.run(tf_matrix_multiplication_prod)


A = tf.get_variable("a_random_int_var_0_to_1", initializer=tf.random_uniform([10000, 10000]))
B = tf.get_variable("b_random_int_var_0_to_1", initializer=tf.random_uniform([10000, 10000]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(A)
sess.run(B)

cpu_timer = timeit.Timer(functools.partial(test,A,B))
cpu_time = cpu_timer.timeit(1)
print(cpu_time)
# 14.6120979786

# GPU time ( SAME code, but using tensorflow gpu enable library ):
#0.46277999300218653