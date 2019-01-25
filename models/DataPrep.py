import numpy as np

def normalize_data(data, mean=None, std=None, skipfields=[], axis=0, skipfields_bool=None):
	'''
	param:
		data: numpy array
		axis: specifies the axis along which data is to be normalized, default is 0
		skipfields: a list with fieldindices which are not to be normalized (they are kept unchanged)
		mean/std are alternative real mean and std of a data distribution which is hidden because 
			the provided data is only a subset. mean and std must have length of axis which is not given in axis
		skipfields_bool: np.bool_, shape [nFields]. If True, skip the field.
			
	returns:
		data: Normalized data with zero mean and standard deviation 1
		mean:
		std:
	'''
	c = False
	try:
		n, p = data.shape
	except ValueError:
		n = len(data)
		p = 1
		c = True
		data = np.reshape(data, (-1,1))
	tiny = 1.e-10
	
	if (mean is None or std is None):
		mean = np.mean(data, axis=axis)
		std = np.std(data, axis=axis)
	else:
		if isinstance(mean, float):
			mean = np.repeat(mean, p)
		if isinstance(std, float):
			std = np.repeat(std, p)
	for field in skipfields:
		mean[field] = 0.
		std[field] = 1.
	if not skipfields_bool is None:
		mean[skipfields_bool] = 0.
		std[skipfields_bool] = 1.
		
	zero_variance = []
	for i, std_ in enumerate(std):
		if std_ < tiny:
			zero_variance.append([i,std_])
			std[i] = 1.
			
	data_ = 1.*(data - mean)/std
	# for i, std_ in zero_variance:
		# std[i] = std_
	
	if c:
		data_ = np.ravel(data_)
		mean = mean[0]
		std = std[0]
		
	return [data_, mean, std]

def fast_random_inputbatch_generator(datax, batch_size, shuffle=True):
	'''
	param:
		datax: array of rank at least one [data_size, ...]
		batch_size: int, specifies size of first dimension of output batch, eg how many samples
		shuffle: bool, default=True
	yields:
		batch: random batch of shape [batch_size, ...]
	'''
	n = len(datax)
	batch_size = min(n, max(1, batch_size))
	if batch_size == n:
		shuffle = False
	
	indices = np.arange(n)
	if shuffle:
		np.random.shuffle(indices)
	i = 0
	while True:

		batchx = datax[indices[i:i+batch_size]]
		i += batch_size
		
		l = len(batchx)
		if l == batch_size:
			yield batchx
			if i == n:
				i = 0
		else:
			if shuffle:
				np.random.shuffle(indices)
			batchx = np.append(batchx, datax[indices[:batch_size - l]], axis=0)
			i = batch_size - l
			yield batchx
