# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
#String operations 

# tensor `a` is [["a", "b"], ["c", "d"]]
import tensorflow as tf

sess = tf.Session()
a = tf.convert_to_tensor([["a", "b"], ["c", "d"]])
print(sess.run(a))

tf.reduce_join(a, 0) 					#==> ["ac", "bd"]
tf.reduce_join(a, 1) 					#==> ["ab", "cd"]
tf.reduce_join(a, -2) 					# ==> ["ac", "bd"]
tf.reduce_join(a, -1) 					# ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) 	#==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) 	#==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") 	#==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) 				#==> ["acbd"]
tf.reduce_join(a, [1, 0]) 				#==> ["abcd"]
tf.reduce_join(a, [])     				#==> ["abcd"]

b = tf.convert_to_tensor(["ac"])
c = tf.convert_to_tensor(["bd"])
d = tf.string_join([b,c], separator=" ", name=None)
print(sess.run(d))

e = tf.reduce_join(a, 0)
print(tf.string_to_hash_bucket(e, 2))
print(sess.run(tf.string_to_hash_bucket(e, 5)))

f = tf.string_to_hash_bucket(e, 2)
hw = tf.convert_to_tensor(["hello worls"])
print(sess.run(tf.string_split(hw, delimiter=' ')))

### Exercise modelue_1_4

#Create new string tensors with:
#	a) transform str_1 in a way to get [["name: ", "surname: "], ["Jan", "Idziak"]]
#   a')str_1 with argument ["name: Jan", "surname: Idziak"] 
#	b) str_2 with argument[["helo ", "world"], ["tensor", "flow"]] 
#	b') str_2 with argument ["helo world","tensorflow"]
#   c) Create simple string tensors with arguments: 
#	c')	str_3 - ["My name is:"]
#	c'') str_4 - ["Janek"] 
#   c''') string_join to obtain ["My name is: Janek"] 
#   c''') string_join to obtain ["My name is:__Janek"] 
#   c''') string_join to obtain ["My name is:randomseparatorJanek"] 
#
#Print results using single initializer 

a = tf.convert_to_tensor([["name: ", "surname: "], ["Jan ", "Idziak "]])
b = tf.convert_to_tensor([["helo ", "world"], ["tensor", "flow"]] )



print sess.run(tf.reduce_join(a, 0))
print sess.run(tf.reduce_join(b, 1))
b = tf.convert_to_tensor(["My name is:"])
c = tf.convert_to_tensor(["Janek"])
d = tf.string_join([b,c], separator=" ", name=None)
print(sess.run(d))
d = tf.string_join([b,c], separator="__", name=None)
print(sess.run(d))
d = tf.string_join([b,c], separator="randomseparator", name=None)
print(sess.run(d))