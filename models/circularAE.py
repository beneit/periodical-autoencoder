# The Autoencoder
from __future__ import print_function

import sys
import os
import time

import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def selu(x, name):
	alpha = 1.6732632423543772848170429916717
	scale = 1.0507009873554804934193349852946
	return tf.multiply(scale,tf.where(x>=0.0, x, alpha*tf.nn.elu(x)), name=name)
	
def swish(x, name, reuse, varName=None):
	if betaName is None:
		betaName = name
	with tf.variable_scope("beta") as scope:
		if reuse:
			scope.reuse_variables()
		beta = tf.get_variable("beta_%s"%varName, shape=[], initializer=tf.ones_initializer())
	return tf.multiply(x,tf.sigmoid(beta*x), name=name)
	

def _get_activation(activation=None):
	if activation == "sigmoid":
		return tf.nn.sigmoid
	elif activation == "relu":
		return tf.nn.relu
	elif activation == "softsign":
		return tf.nn.softsign
	elif activation == "tanh":
		return tf.nn.tanh
	elif activation == "linear" or activation == "lin" or activation is None:
		return tf.identity
	elif activation == "elu":
		return tf.nn.elu
	elif activation == "selu":
		return selu
	elif activation == "softmax":
		return tf.nn.softmax
	elif activation == "swish":
		return swish
	elif activation == "sin":
		return tf.sin
	else:
		raise ValueError("Unknown activation '%s'" % activation)
	

def _get_optimizer(optimizer, learning_rate=0.001, momentum=0.1, name=None):
	if name is None:
		name = optimizer
	if optimizer == "Adam":
		return tf.train.AdamOptimizer(learning_rate, name=name)
	elif optimizer == "Adagrad":
		return tf.train.AdagradOptimizer(learning_rate, name=name)
	elif optimizer == "Adadelta":
		return tf.train.AdadeltaOptimizer(learning_rate, name=name)
	elif optimizer == "Momentum":	
		return tf.train.MomentumOptimizer(learning_rate, momentum, name=name)
	elif optimizer == "GradientDescent":
		return tf.train.GradientDescentOptimizer(learning_rate, name=name)
	else:
		raise ValueError("Unknown optimizer '%s'" % optimizer)
	

class Autoencoder(object):
	"""
	Asymmetric autoencoder, but layers are built symmetrically only.
	A symmetric AE shares the weights in encoding and decoding stage, while an asymmetric AE
	does not. Choose normalization if desired or implement your own in the if, then elif, ... question
	norm such as "Layer" or "Batch" normalization. Also possible would be extensions like residual nets
	(not implemented). Training proceeds as whole neural net. A layer wise training is not possible.
	"""
	
	loss_log = []
	min_loss = 1.e99
	increase_counter = 0
	save = False
	save_counter = 0
	nLayer = None ## def in __init__ method !!!
	
	def _get_number(self, last=False):
		nn = 1
		while os.path.isdir("tmp/summary_"+self.flag_name+("_%d" % nn)):
			nn += 1
		if last:
			return nn - 1
		else:
			return nn
			
	def _fully_connected_stack(self, x, weightNames, biasNames, layers, activation_fn, initializer_bias,
		initializer_weights, aname, norm="None", layer_norm_name=None, reuse=False, transpose=False,
		scope_name="trainable/encoder"):
		"""
		Args:
			x: tensor, input
			weightNames, biasNames: list of string, names for the variables, same size as layers
			layers: array, not including input size
			activation_fn: list of tensorflow function. Is applied to each layer, excluding the input, including output
				(default: None, which is a linear activation) same size as layers
			aname: names which is given to the activations, same size as layers
			norm: "None", "Layer"
			layer_norm_name: list of names for the layer norm scope, same size as layers
			reuse: whether to reuse the variables
			transpose: whether to transpose the weight variables
		Returns:
			a: tensor, the activation of the last layer
		"""
		x_shape = x.shape[1]
		i = 0
		a = x

		for i, layer in enumerate(layers):

			l_ = x_shape if i == 0 else layers[i-1] # previous size, shape for the weights

			shape = [l_, layer] if not transpose else [layer, l_]
			with tf.variable_scope(scope_name) as scope:
				if reuse:
					scope.reuse_variables()
				b = tf.get_variable(biasNames[i], shape=[layer], initializer=initializer_bias)
				if transpose and not reuse:
					scope.reuse_variables()
				w = tf.get_variable(weightNames[i], shape=shape, initializer=initializer_weights)
				if transpose:
					w = tf.transpose(w)
				
				
			if activation_fn[i] is not None:
				a = tf.matmul(a, w)

				# Implement normalization or other extra operations
				if norm == "Layer":
					with tf.variable_scope(scope_name) as scope:
						with tf.variable_scope(layer_norm_name[i]) as layer_norm_scope:
							a = tf.contrib.layers.layer_norm(a, scope=layer_norm_scope, reuse=reuse)

				a = tf.add(a, b)
				try:
					a = activation_fn[i](a, name=aname[i])
				except TypeError:
					a = activation_fn[i](a, name=aname[i], reuse=reuse, varName=weightNames[i])
			else:
				a = tf.add(tf.matmul(a, w), b, name=aname[i])
		return a
	
	def __del__(self):
		if self.save:
			save_path = self.saver.save(self.sess, self.save_path)
			print("Saved neural net in: %s" % save_path)
		tf.reset_default_graph()
	def __init__(self, import_dir=None, nDim_high=None, nDim_low=None, layers=[],
		activations="tanh", encoder_activations=None, decoder_activations=None,
		last_layer="linear", norm="None",
		uniform_bias=False, uniform_weights=False, factor_bias=0.0, factor_weights=1.0,
		mode_bias='FAN_AVG', mode_weights='FAN_AVG', 
		optimizer="Adam", learning_rate=.001, momentum=.1, objective="L2", 
		load=False, number=None, dtype="float", tensorboard=True):
		"""
		Args:
			import_dir: string, path to directory which contains the saved model, all other arguments will be ignored except number and tensorboard
			nDim_high: number of input dimension of the data
			nDim_low: desired number of intermediate lowest dimensional representation
			layers: list of int. Size of input, all hidden and encoding layer. Specify this or nDim_high/nDim_low, not both.
			activations: to be used: relu, sigmoid, tanh, softsign, elu, selu, softmax, swish
			encoder_activation: activation function in the encoding layer, see activations
			last_layer: activation function for the last layer if given. The last layer has same size as output and performs one last transformation.
				Put None if you do not wish to have a such last layer
			norm: additional normalization; "None", "Layer", 
			weight/bias initialization parameter, see tf.contrib.layers.variance_scaling_initializer
				uniform, factor, mode (default is xavier initialization, set to 1, FAN_IN to self normalizing initialization
			optimizer: GradientDescent, Adam, Adagrad, Adadelta, Momentum
			learning_rate: the learning rate
			objective: the objective which is to be minimized, specify
				L2: quadratic distance between in/- and output
				CrossEntropy
			load: whether to load this model or create new one (path is standardized)
			number: which number to give the net, will give new number if None
			dtype: data type to be used in tensorflow, "float" or "double"
			tensorboard: whether to use the tensorboard
		"""
		
		if import_dir is not None:
			self.sess = tf.Session()
			self.sess.as_default()
			tf.saved_model.loader.load(self.sess, ["main"], import_dir)
			
			self.nLayer = self.sess.run("nLayer:0")
			self.layers = self.sess.run("layers:0")
			
			self.init = tf.global_variables_initializer()
		else:
			
			if dtype=="float":
				dtype = tf.float32
			else:
				dtype = tf.float64
			
			if not os.path.isdir("out"):
				os.makedirs("out")
			if not os.path.isdir("out/vid"):
				os.makedirs("out/vid")
			if tensorboard and not os.path.isdir("tmp"):
				os.makedirs("tmp")
				
			
			# self saves: layer, number, optimizer, ...?
			self.tensorboard = tensorboard
			
			# layer
			layers = [nDim_high] + list(layers)
			self.nLayer = len(layers) - 1
			self.layers = layers
			put_last_layer = last_layer is not None
			if not encoder_activations is None and len(encoder_activations) != self.nLayer:
				raise ValueError("Given length of encoder activations is invalid: {actlength} for size of {size}".format(actlength=len(encoder_activations), size=self.nLayer))
			if not decoder_activations is None and len(decoder_activations) != self.nLayer:
				raise ValueError("Given length of decoder activations is invalid: {actlength} for size of {size}".format(actlength=len(decoder_activations), size=self.nLayer))
				
			
			
			tf.constant(put_last_layer, name="put_last_layer")
			if not put_last_layer:
				last_layer = "None"
			tf.constant(last_layer, name="last_layer")
			tf.constant(self.nLayer, name="nLayer") # Use this one to def nLayer when loading a tensorflow model: self.nLayer = sess.run("nLayer:0")
			tf.constant(self.layers, name="layers") # Use this one to def layers when loading a tensorflow model: self.layers = sess.run("layers:0")
			tf.constant(activations, name="activations")
			tf.constant(norm, name="norm")
			
			self.flag_name = ("cAE" + "_%d"*len(layers)) % tuple(layers)
			self.save_path = "out/"+self.flag_name+".ckpt"
			print("Encoder network layers (inclusive in/- and output): ", end="")
			print(layers + [1])
			

		
			# neural net: input
			x = tf.placeholder(dtype, shape=[None,nDim_high], name="x")
			y = tf.placeholder(dtype, shape=[None,nDim_low], name="y") # this is an "encoding" custom input node
			
			
			freq = tf.constant([[1,1]], name='freq', dtype=dtype)
			phase = tf.get_variable('trainable/phase', initializer=[np.pi/2,0], trainable=False)
			if nDim_low == 1:
				pass
			elif nDim_low == 2:
				capR = tf.get_variable('trainable/capR', dtype=tf.float32, initializer=[1.,1.], trainable=True)
			else:
				raise ValueError("Currently only nDim_low = 1 allowed!")

			initializer_bias = lambda: tf.contrib.layers.variance_scaling_initializer(factor=factor_bias, mode=mode_bias, uniform=uniform_bias)
			initializer_weights = lambda: tf.contrib.layers.variance_scaling_initializer(factor=factor_weights, mode=mode_weights, uniform=uniform_weights)
			
			# encoder:
			intermediate_layers = layers[1:]
			weightNames = ["encoder_weights_d%d"%d for d in range(self.nLayer)]
			biasNames = ["encoder_bias_d%d"%d for d in range(self.nLayer)]
			activation_fn = [_get_activation(activations) for _ in range(self.nLayer)] if encoder_activations is None else [_get_activation(encoder_activations[d]) for d in range(self.nLayer)]
			aname = ["encoder_act_d%d"%(d) for d in range(self.nLayer)]
			layer_norm_name = ["encoder_layer_norm_d%d"%d for d in range(self.nLayer)]
			
			enc = self._fully_connected_stack(x, weightNames, biasNames, intermediate_layers, activation_fn,
					initializer_bias(), initializer_weights(), aname, norm=norm, layer_norm_name=layer_norm_name, reuse=False, transpose=False)
			
			# tanh transformations
			wx = tf.get_variable('trainable/encoder/wx', shape=[layers[-1],nDim_low + 1], initializer=initializer_weights())
			bx = tf.get_variable('trainable/encoder/bx', shape=[nDim_low + 1], initializer=initializer_bias())
			xxx = tf.add(tf.matmul(enc, wx), bx, name="xxx")
			if nDim_low == 1:
				phi = tf.reshape(tf.atan2(xxx[:,1], xxx[:,0]), [-1, 1], name='phi')
			if nDim_low == 2:
				# theta = tf.asin(tf.nn.sigmoid(xxx[:,2]))
				theta = xxx[:,2]
				cosTheta = tf.cos(theta)
				sinTheta = tf.sin(theta)
				phi1 = tf.reshape(tf.atan2(xxx[:,1]/(1 + cosTheta), xxx[:,0]/(1 + cosTheta)), [-1,1])
				
				phi = tf.concat([tf.reshape(theta, [-1,1]), phi1], axis=1, name='phi')
			# decoder:				
			# circularity
			if nDim_low == 1:						
				circ = tf.sin(tf.matmul(phi, freq) + phase, name="circ")
				circ_custom = tf.sin(tf.matmul(y, freq) + phase, name="circ_custom")
			if nDim_low == 2:
				xback_yback = (capR + tf.reshape(tf.tile(cosTheta, [2]), [-1,2]))*tf.sin(tf.matmul(phi1, freq) + phase)
				circ = tf.transpose(tf.stack([xback_yback[:,0], xback_yback[:,1], sinTheta], name="circ"))
				
				# custom
				sinTheta_custom = tf.sin(y[:,1])
				cosTheta_custom = tf.cos(y[:,1])
				xback_yback_custom = (capR + tf.reshape(tf.tile(cosTheta_custom, [2]), [-1,2]))*tf.sin(tf.matmul(tf.reshape(y[:,0], [-1,1]), freq) + phase)
				circ_custom = tf.transpose(tf.stack([xback_yback_custom[:,0], xback_yback_custom[:,1], sinTheta_custom], name="circ_custom"))

			# rest
			intermediate_layers = list(layers[self.nLayer - 1::-1])
			wName = "decoder"
			weightNames = ["%s_weights_d%d"%(wName,d) for d in range(self.nLayer - 1 + put_last_layer, -1, -1)]
			biasNames = ["decoder_bias_d%d"%d for d in range(self.nLayer - 1 + put_last_layer, -1, -1)]
			activation_fn = [_get_activation(activations) for d in range(self.nLayer - 1, -1, -1)] if decoder_activations is None else [_get_activation(decoder_activations[d]) for d in range(self.nLayer)]
			aname = ["decoder_act_d%d"%(d) for d in range(self.nLayer - 1 + put_last_layer, -1, -1)]
			layer_norm_name = ["decoder_layer_norm_d%d"%d for d in range(self.nLayer - 1 + put_last_layer, -1, -1)]
			scope_name="trainable/%s"%wName
			
			if put_last_layer:
				intermediate_layers.append(layers[0])
				activation_fn.append(_get_activation(last_layer))

			dec = self._fully_connected_stack(circ, weightNames, biasNames, intermediate_layers, activation_fn,
					initializer_bias(), initializer_weights(), aname, norm=norm, layer_norm_name=layer_norm_name, reuse=False, transpose=False, scope_name=scope_name)
			
			# Now build one additional FULL path in the case where the user wants the decoding of his own latent space points
			aname = ["decoder_act_custom_d%d"%(d) for d in range(self.nLayer - 1 + put_last_layer, -1, -1)]
			decoder_custom = self._fully_connected_stack(circ_custom, weightNames, biasNames, intermediate_layers, activation_fn,
						None, None, aname, norm=norm, layer_norm_name=layer_norm_name, 
						reuse=True, transpose=False, scope_name=scope_name)

					
			# Quadratic loss, start learning from p=0
			l2_loss = tf.reduce_mean(tf.square(x - dec), name="l2_loss")
			# Reconstruction error
			l2_loss_single = tf.reduce_mean(tf.square(x - dec), axis=1, name="l2_loss_single")
			l2_loss_single_feature = tf.square(x - dec, name="l2_loss_single_feature")
			
			
			# the optimizer, one for normal and one for the encoder
			optimizerOp = _get_optimizer(optimizer, learning_rate, momentum)
			optimizerEnc = _get_optimizer(optimizer, name=optimizer + "_encoder")
			
			
			# the train step, one for each optimizer
			train = optimizerOp.minimize(l2_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), name="train")
			train_enc = optimizerEnc.minimize(l2_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="trainable/encoder"), name="train_encoder")
			
			# Session and initialization
			self.sess = tf.Session()
			self.sess.as_default()
			self.init = tf.global_variables_initializer()
			

			# Tensorboard
			if tensorboard:
				# merged_p = []
				tf.summary.scalar("l2_loss", l2_loss)
				# Variable summaries
				# tf.summary.histogram("w_encoder_%d" % p, self.variables[2*p,0])
				# tf.summary.histogram("b_encoder_%d" % p, self.variables[2*p,1])
				# tf.summary.histogram("w_decoder_%d" % p, self.variables[2*p+1,0])
				# tf.summary.histogram("b_decoder_%d" % p, self.variables[2*p+1,1])
				# Activation summaries
				# Encoder
				if nDim_low == 1:
					tf.summary.histogram("a_encoder", phi)
				# Decoder
				
				# tf.summary.scalar("cross_entropy_%d" % p, self.cross_entropy[p])
				#self.merged_p.append(tf.summary.merge(["l2_loss_%d" % p]))
				pass
				merged = tf.summary.merge_all()

				if number is None and load:
					self.number = self._get_number(last=False)
				elif number is None:
					self.number = self._get_number()
				twriter_path = os.path.join(os.getcwd(), "tmp", "summary_" + self.flag_name + ("_%d" % self.number))
				self.test_writer = tf.summary.FileWriter(twriter_path, self.sess.graph)
				print("To use tensorboard, type:")
				print("tensorboard --logdir='" + twriter_path + "'")
			
			# Save and Load
			self.saver = tf.train.Saver()
			if load:
				self.saver.restore(self.sess, self.save_path)
			else:
				# Initialize all variables
				self.sess.run(self.init)
				
				
	def _validate(self, test_data, i, print_console=True, early_stopping=False, plot=False, train_loss=0.):
		
		if test_data is not None:
			if self.tensorboard:
				summary, al2_loss, proj = self.sess.run(["Merge/MergeSummary:0", "l2_loss:0", "encoder_act_d%d:0"%(self.nLayer - 1)], feed_dict={"x:0": test_data})
				self.test_writer.add_summary(summary, i)
			else:
				al2_loss, proj = self.sess.run(["l2_loss:0", "encoder_act_d%d:0"%(self.nLayer - 1)], feed_dict={"x:0": test_data})
		else:
			al2_loss = train_loss
		self.loss_log.append([i, al2_loss, train_loss])


		
		if plot:
			proj = proj.transpose()
			plt.close("all")
			print(proj[0].shape)
			print(proj[1].shape)
			print(al2_loss.shape)
			plt.scatter(proj[0], proj[1], c=al2_loss, s=8, lw=0)
			plt.savefig("out/vid/dim_low_"+self.flag_name+"_scatter.png", dpi=150)
			# else: # if we want to plot with nDim_high > 2
				# plotroutines.draw_triangle_cont(proj, al2_loss, plotname="out/vid/dim_low_"+self.flag_name+"_scatter.png") 
				
		if early_stopping and test_data is not None:
			if al2_loss < self.min_loss:
				if self.save_counter < 10 or self.save_counter%10 == 0:
					self.min_loss = al2_loss
					save_path = self.saver.save(self.sess, "out/"+self.flag_name+"_tmp.ckpt")
				self.save_counter += 1
			
			if len(self.loss_log) > 1 and al2_loss > self.loss_log[-2][1]:
				self.increase_counter += 1
				self.save_counter = 0
			else:
				self.increase_counter = 0
		
		if print_console:
			print("Step %d:  l2loss = %.5g   Train: %.5g" % (i, al2_loss, train_loss))
		# self._print_weights()
		
	def train(self, nTrain, batch_gen, test_data, save=False, test_step=500, print_step=5000, count_tests=False, early_stopping=False, max_increase_steps=10, plot=False):
	
		self.save = save
		t0 = time.time()
		
		l2_loss_ = 0	
		step = 0
		while step < nTrain:
			batch = next(batch_gen)
			
			l2_loss_, _ = self.sess.run(["l2_loss:0", "train"], feed_dict={"x:0": batch})
			if (step % test_step == 0):
				tx = step/test_step if count_tests else step
				self._validate(test_data, tx, (step % print_step == 0), early_stopping=early_stopping, plot=plot, train_loss=l2_loss_)
				if self.increase_counter >= max_increase_steps:
					break
			step += 1
		tx = step/test_step if count_tests else step
		self._validate(test_data, tx, early_stopping=early_stopping, plot=plot, train_loss=l2_loss_)
		self.loss_log = np.array(self.loss_log)
		
		if save:
			save_path = self.saver.save(self.sess, self.save_path)
			
		print("Training time: %.3f" % (time.time()-t0))
		return step
		
	def restore_checkpoint(self, checkpoint=None):
		if checkpoint is None:
			checkpoint = "out/"+self.flag_name+"_tmp.ckpt"
		try:
			self.saver.restore(self.sess, checkpoint)
		except Exception as e:
			print("Error: Could not load checkpoint '%s'. Possible reasons: Invalid architecture, invalid source."%checkpoint)
			print(e)
			return 1
		return 0
		
		
	def get_encoder_weights(self, layer=0):
		return self.sess.run("trainable/encoder/encoder_weights_d%d:0"%layer)
	def get_decoder_weights(self, layer=0):
		return self.sess.run("trainable/encoder/decoder_weights_d%d:0"%layer)
		
	def _print_weights(self):
		# return 0
		print("______VARIABLES_____")
		for d in range(self.nLayer):
			w = self.sess.run("trainable/encoder_weights_d%d:0"%d)
			print(w.sum(), end="\t")
			w = self.sess.run("trainable/decoder_weights_d%d:0"%d)
			print(w.sum(), end="\t")
		print("\n__END_VARIABLES______")
		
	def _batch_process_x(self, tensor, x, batch_size=-1, y=False):
		if batch_size is None or batch_size < 0 or batch_size > len(x):
			batch_size = len(x)
		if isinstance(tensor, list):
			N = len(tensor)
		else:
			N = 0			
		if N == 0:
			result = []
		else:
			result = [[] for _ in range(N)]
			
		if y:
			feed_in = "y:0"
		else:
			feed_in = "x:0"
			
		i = 0
		while True:
			high = i + batch_size if i + batch_size < len(x) else len(x)
			tmp = self.sess.run(tensor, feed_dict={feed_in: x[i:high]})
			if N == 0:
				result.append(tmp)
			else:
				for t in range(N):
					result[t].append(tmp[t])
			
			i += batch_size
			if high == len(x):
				break
		if N == 0:
			return np.concatenate(result, axis=0)
		else:
			return [np.concatenate(result_, axis=0) for result_ in result]
		
		
	def project_data(self, data, batch_size=None):
		"""
		Returns:
			Encoding, shape=[N,nDim_low]
			Decoding, shape=[N,nDim_high]
			L2Loss per datum, shape=[N]
		"""
		return self._batch_process_x(["phi:0", "decoder_act_d%d:0"%(0), "l2_loss_single:0"], data, batch_size)
		
	def project_data_single(self, data, batch_size=None):
		"""
		Returns:
			Encoding, shape=[N,nDim_low]
			Decoding, shape=[N,nDim_high]
			L2Loss per datum and feature, shape=[N,nDim_high]
		"""
		return self._batch_process_x(["encoder_act_d%d:0"%(self.nLayer - 1), "decoder_act_d%d:0"%(0), "l2_loss_single_feature:0"], data, batch_size)
		
	def reconstruct(self, data, batch_size=None):
		"""
		Returns:
			L2Loss per datum, shape=[N]
		"""
		return self._batch_process_x("l2_loss_single:0", data, batch_size)
		
	def reconstruct_single(self, data, batch_size=None):
		"""
		Returns:
			L2Loss per datum and feature, shape=[N,nDim_high]
		"""
		return self._batch_process_x("l2_loss_single_feature:0", data, batch_size)
		
	def encode(self, data, batch_size=None):
		"""
		Returns:
			Encoding, shape=[N,nDim_low]
		"""
		return self._batch_process_x("encoder_act_d%d:0"%(self.nLayer - 1), data, batch_size)
		
	def decode(self, data, batch_size=None):
		"""
		Args:
			Data in the latent space, shape=[N,nDim_low]. if nDim_low==1, then it is the polar angle. if nDim_low==2, then it is (phi, theta) in torus coordinates. phi revolves in x-y-plane.
		Returns:
			Decoding, shape=[N,nDim_high]
		"""
		put_last_layer = self.sess.run("put_last_layer:0")
		return self._batch_process_x("decoder_act_custom_d%d:0"%(0), data, batch_size, y=True)
		
	def save_loss_log(self, plot=True):
		# return 0
		np.save("out/losses_"+self.flag_name + ("_%d" % self.number) + ".npy", self.loss_log)
		
		if plot:
			# LOGPLOTS
			plt.close("all")
			#plt.plot(self.loss_log[:][0], self.loss_log[:][1], color="red")
			plt.semilogy(self.loss_log[:,0], self.loss_log[:,1], color="red")
			axes = plt.gca()
			axes.set_ylim([np.amin(self.loss_log[:,1]),1.5*np.mean(self.loss_log[:,1])])
			plt.xlabel("step")
			plt.ylabel("l2_loss")
			plt.savefig("out/l2_loss_log_" + self.flag_name + ("_%d" % self.number) + ".png")
			
			if False:
				plt.close("all")
				plt.semilogy(self.loss_log[:,0], self.loss_log[:,2], color="blue")
				axes = plt.gca()
				axes.set_ylim([np.amin(self.loss_log[:,2]),1.5*np.mean(self.loss_log[:,2])])
				plt.xlabel("step")
				plt.ylabel("cross_entropy")
				plt.savefig("out/cross_entropy_log_" + self.flag_name + ("_%d" % self.number) + ".png")
			
			# NORMAL PLOTS
			plt.close("all")
			plt.plot(self.loss_log[:,0], self.loss_log[:,1], color="red")
			axes = plt.gca()
			axes.set_ylim([np.amin(self.loss_log[:,1]),1.5*np.mean(self.loss_log[:,1])])
			plt.xlabel("step")
			plt.ylabel("l2_loss")
			plt.savefig("out/l2_loss_" + self.flag_name + ("_%d" % self.number) + ".png")
			
			if False:
				plt.close("all")
				plt.plot(self.loss_log[:,0], self.loss_log[:,2], color="blue")
				axes = plt.gca()
				axes.set_ylim([np.amin(self.loss_log[:,2]),1.5*np.mean(self.loss_log[:,2])])
				plt.xlabel("step")
				plt.ylabel("cross_entropy")
				plt.savefig("out/cross_entropy_" + self.flag_name + ("_%d" % self.number) + ".png")
				
	def save_model(self, export_dir):
		"""
		Use method to save the complete model, which is graph and variables. If you want to load, initialize with the
		export_dir as the import_dir and nothing else.
		Alternatively you can load from a checkpoint set after training, by setting save=True flag in the initialization.
		The Checkpoint only saves the variables though, so the graph will need to be rebuilt.
		"""
		builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
		builder.add_meta_graph_and_variables(self.sess, ["main"])
		builder.save()