#!/usr/bin/python3.8

#from ML_tuning import CLIC_ML
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
from tensorflow import keras

from multiprocessing import Pool

import numpy as np
import pandas as pd

import pickle as pk
import copy

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

import warnings

warnings.simplefilter(action = 'ignore', category=UserWarning)

def TrainModel(features_archive, weights_archive, quad_to_use, folder_in):
	

	quad_to_scan = quad_to_use


	features = features_archive + [quad_to_scan]

	n_features = len(features)

	filter_zeros = lambda data: list(filter(lambda x: abs(x) != 0.0, data))

	error_scale = tf.constant(1.0, dtype = tf.float32)

	BOTTOM_LIMIT = tf.constant([1.0], dtype = tf.float32)
	UPPER_LIMIT = tf.constant([100.0], dtype = tf.float32)

	zero = tf.constant([0.0], dtype = tf.float32)

	error_scale = tf.constant(5e-7, dtype = tf.float32)# before was 1e-4
	orthogonality_error_scale = tf.constant(1e-1, dtype = tf.float32)

	orbit_error_scale = tf.constant(1e-6, dtype = tf.float32)

	exit_orbit_error_scale = tf.constant(1e-8, dtype = tf.float32)

#	ORBIT_LIMIT = tf.constant(20.0, dtype = tf.float32) # 20 microns limit
	ORBIT_LIMIT = tf.constant(40.0, dtype = tf.float32) # 40 microns limit

	N_LAST_BPMS_TO_FLATTEN = 2

	

	class WeightsClipper3(keras.constraints.Constraint):

		def __init__(self, upper_bound):
			self.upper_bound = upper_bound

		def __call__(self, w):
			return tf.map_fn(self._clip, w)

		def _clip(self, weight):
			# outside upper boundary
			if tf.math.less(weight, -self.upper_bound):
				return -self.upper_bound
			if tf.math.greater(weight, self.upper_bound):
				return self.upper_bound

			return weight

		def get_config(self):
			return {'upper_bound': self.upper_bound}


	R_adj = np.eye(2063)
#	print(R_adj)
	factor = 10.0
	for i in range(1489):
		R_adj[i, i] = factor
	# Reading the data
	#R = pk.load(open("response_matrix.pkl", "rb"))
	with open("data/knobs/S11_M5_opt_knobs_after_rf/response_matrix_fixed.pkl", "rb") as response_file:
		R = pk.load(response_file)

	with open("data/knobs/S11_M5_opt_knobs_after_rf/orbit_response.pkl", "rb") as orbit_file:
		R_orbit = pk.load(orbit_file)

	# Target matrix
	y = np.eye(110)
	y_index = 3 #corresponds to the Y4 knob
	Y = y[:, y_index] * 1e-2
	Y = np.reshape(Y, (110, 1))
#	print(Y)

	# reading the orbit response matrix


	R_adjusted = R.dot(R_adj)
	R_orbit_adjusted = R_orbit.dot(R_adj)

	R_adjusted_cut = R_adjusted[:69, :]
	Y_cut = Y[:69]
	#arguments_sorted = pk.load(open(f"data/learning_storage/orbit_supression_2/Y{y_index + 1}_postcheck1/features_importance.pkl", "rb"))

#	FOLDER = f"data/learning_storage/orbit_supression_2/Y{y_index + 1}_sfs_test2_{n_features}_quads"
	FOLDER = folder_in

	# reading the orbit response matrix
#	R_orbit = pk.load(open("data/knobs/S11_M5_opt_knobs_after_rf/orbit_response.pkl", "rb"))

	R_adjusted = R.dot(R_adj)
	R_orbit_adjusted = R_orbit.dot(R_adj)

	R_adjusted_cut = R_adjusted[:69, :]
	Y_cut = Y[:69]

	def cut_data4(features):

		return features, R_adjusted_cut[:, features], R_orbit_adjusted[:, features]


	def reg_orbit_penalty(weights):

		convertion_factor = tf.constant(1e4, dtype = tf.float32) # x 1e6 (to have microns) * 1e-2 (for weights)
		convertion_factor = tf.constant(1e6, dtype = tf.float32) # x 1e6 (to have microns)

		orbit_vector = tf.linalg.matvec(R_orbit_tensor, tf.squeeze(weights, axis=-1)) * convertion_factor
		
		filtered_vector = tf.where(abs(orbit_vector) > ORBIT_LIMIT, abs(orbit_vector) - ORBIT_LIMIT, tf.zeros_like(orbit_vector))

		orbit_penalty = tf.reduce_mean(filtered_vector)
		
		return orbit_penalty * orbit_error_scale
	
	def reg_exit_orbit_penalty(weights):

		convertion_factor = tf.constant(1e4, dtype = tf.float32) # x 1e6 (to have microns) * 1e-2 (for weights)
		convertion_factor = tf.constant(1e6, dtype = tf.float32) # x 1e6 (to have microns)

		orbit_vector = tf.linalg.matvec(R_orbit_tensor, tf.squeeze(weights, axis=-1)) * convertion_factor
		
		last_values = orbit_vector[-N_LAST_BPMS_TO_FLATTEN:]
		exit_orbit_penalty = tf.reduce_sum(tf.abs(last_values))

		return exit_orbit_penalty * exit_orbit_error_scale

	def reg_zero_penalty(weights):

		filtered_weights = tf.where(tf.abs(weights) < BOTTOM_LIMIT, tf.abs(weights) - BOTTOM_LIMIT, tf.zeros_like(weights))

		zero_penalty = tf.abs(tf.reduce_mean(filtered_weights))

		return error_scale * zero_penalty

	def regularizer_mod_5_3(weights):
		"""
		same as regularizer_mod_2() but with the orbit suppression
		"""
#		print(weights)

		zero_penalty = reg_zero_penalty(weights)
		orbit_penalty = reg_orbit_penalty(weights)
#		print(f"Zero penalty = {zero_penalty}, orbit penalty = {orbit_penalty}")

		return zero_penalty + orbit_penalty
	
	def regularizer_mod_5_4(weights):
		"""
		same as regularizer_mod_5_3() but with exit orbit supression
		"""
#		print(weights)

		zero_penalty = reg_zero_penalty(weights)
		orbit_penalty = reg_orbit_penalty(weights)
		exit_orbit_penalty = reg_exit_orbit_penalty(weights)
#		print(f"Zero penalty = {zero_penalty}, orbit penalty = {orbit_penalty}")

		return zero_penalty + orbit_penalty + exit_orbit_penalty


	early_stopping = EarlyStopping(monitor = 'loss', patience = 1000, mode = 'min', min_delta = 1e-10, verbose = 2)
	early_stopping2 = EarlyStopping(monitor = 'loss', patience = 100000, mode = 'min', min_delta = 1e-7, verbose = 2)

	weights_clipper = WeightsClipper3(UPPER_LIMIT)

	model = keras.Sequential()
	#    model.add(keras.layers.Dense(1, use_bias = False, input_shape = (n_features,), kernel_regularizer = regularizer2))
	optimizer = keras.optimizers.Adam(learning_rate = 0.1)

	new_weights = np.concatenate((weights_archive, np.array([0.0])))
	initializer = tf.keras.initializers.Constant(new_weights)

	model.add(keras.layers.Dense(1, use_bias = False, input_shape = (len(features),), kernel_constraint = weights_clipper, 
							kernel_initializer = initializer, kernel_regularizer = regularizer_mod_5_4))

	model.compile(loss = 'mean_squared_error', optimizer = optimizer, run_eagerly = False)

	features_id, R_cut, R_orbit_cut = cut_data4(features)
	#features_id, R_cut, R_orbit_cut = range(n_features), R_adjusted_cut, R_orbit_adjusted

	R_orbit_tensor = tf.constant(R_orbit_cut, dtype = tf.float32)

	history = model.fit(tf.convert_to_tensor(R_cut, dtype = tf.float32), tf.convert_to_tensor(Y_cut, dtype = tf.float32), 
					epochs = 1000000, callbacks = [early_stopping, early_stopping2], 
					batch_size = 69, verbose = 0)
	
	weights = model.get_weights()[0]

	#	score = r2_score(Y, R_cut @ weights)
	score = r2_score(Y_cut, R_cut @ weights)

	nonzero_elems = filter_zeros(weights.ravel())

	tmp = {
		'Regularization': "regularizer_mod_5_3", 
		'alpha': float(error_scale),
		'n_features': n_features,
		'features_ids': features_id,
		'n_nonzero_elems': len(nonzero_elems), 
		'nonzero_elems': nonzero_elems, 
		'score': score, 
		'max_weight': float(max(tf.math.abs(weights))), 
		'min_weight': float(min(tf.math.abs(weights))),
		'weights': weights,
		'total_loss': history.history['loss'][-1],
		'zero_loss': float(reg_zero_penalty(tf.constant(weights, dtype = tf.float32))),
		'orbit_loss': float(reg_orbit_penalty(tf.constant(weights, dtype = tf.float32))),
		'exit_orbit_loss': float(reg_exit_orbit_penalty(tf.constant(weights, dtype = tf.float32)))
	}

	with open(os.path.join(FOLDER, f"quad_{quad_to_scan}.pkl"), 'wb') as summary_file:
		pk.dump(tmp, summary_file)

def gather_data(folder):
	data = []
	_start, _end = 1489, 2062

	feature_id = _start

	print("Found ", end = "")
	for feature_id in range(_start, _end + 1):
		_filename = f"quad_{feature_id}.pkl"
		if os.path.isfile(os.path.join(folder, _filename)):
			print(feature_id, end = ", ")
			with open(os.path.join(folder, _filename), 'rb') as file:
				row_data = pk.load(file)
				
				data.append(row_data)

	print()
	res = pd.DataFrame(data)
	return res

def gather_data2(folder):
	data = []
	_start, _end = 0, 2062

	feature_id = _start

	print("Found ", end = "")
	for feature_id in range(_start, _end + 1):
		_filename = f"quad_{feature_id}.pkl"
		if os.path.isfile(os.path.join(folder, _filename)):
			print(feature_id, end = ", ")
			with open(os.path.join(folder, _filename), 'rb') as file:
				row_data = pk.load(file)
				
				data.append(row_data)

	print()
	res = pd.DataFrame(data)
	return res

def build_zip_parameters(features_used, weights_used, parameters, folder):
    """
    Create a zipped list for the initiating of TrainModel

    If features_used does not include quads (no number in the 1489 - 2062) 
    than any quad that we add (in the range 1489-2060) we includeit  along with the other 2 
    quads (in particular 2061, 2062).

    If features_used contains quad - we add the other quad like usual
    """

    # determining if the quads are already in use
    quad_in_the_list = False

    for feature in features_used:
        if feature in range(1489, 2063):
            quad_in_the_list = True
    
    print(quad_in_the_list, end = ", ")
    
    features_used_w_exit_quads = copy.copy(features_used)
    weights_used_w_exit_quads = copy.copy(weights_used)
    
    features_used_w_exit_quads.extend([2061, 2062])
    weights_used_w_exit_quads.extend([0.0, 0.0])
    
    if quad_in_the_list:
        return zip([features_used] * len(parameters), [weights_used] * len(parameters), parameters, [folder] * len(parameters))
    else:
        features_list, weights_list = [], []
        # quads are not used yet
        for i in range(len(parameters)):
            #print(i, features_list)
            if parameters[i] in range(1489, 2061):
                features_list.append(features_used_w_exit_quads)
                weights_list.append(weights_used_w_exit_quads)
            else:
                features_list.append(features_used)
                weights_list.append(weights_used)
        return zip(features_list, weights_list, parameters, [folder] * len(parameters))


# Initializing the last 2 quads to be used primaraly to flatten the exit
# beam orbit
#optimal_features = [2061, 2062, 1565, 1777, 1535, 2007, 1531, 1587, 1773, 1550, 1984, 1919, 1661, 1715, 1598, 1884, 1645, 2059, 1866, 1575, 1878, 1513, 1755, 2058, 1611, 1806]

#optimal_weights = [-34.88219, 12.352649, -1.0037875, -29.490622, 4.008907, 21.364164, 4.7346954, 1.0026344, -2.266208, 1.0030671, 10.4665575, -7.2849536, -29.036022, 31.695919, -4.3272247, 1.0018942, 15.973233, 29.884201, 7.8011074, -5.377925, 5.283347, 1.9028009, 1.0011598, 12.834574, 11.793039, 1.0032954]

optimal_features, optimal_weights = [], []

#optimal_features = [90, 57, 376, 234, 8, 2061, 2062, 2047, 1136, 119]
#optimal_weights = [-14.190321, -5.097989, -24.316925, -20.416763, 2.4696467, -1.0023935, 15.024599, -1.0965123, -43.110172, -11.954066]

y_index = 3

iteration_start_index = 0

#for i in range(20):
for i in range(iteration_start_index, 20):

	# Define the different parameters you want to use
	parameters = list(range(1489, 2061))

	# Quads summary
	#features_used = [573, 90, 115, 376, 8, 47, 368, 158, 564, 3, 207, 30, 530, 70, 0, 28]

	#features_used = [2062, 1611, 1657, 1583, 2061, 1805, 2058, 1909, 1859, 1587, 2059, 1932, 1491, 2043, 1897, 2055, 1771]

	features_used = optimal_features
	weights_used = optimal_weights

	folder = f"data/learning_storage/orbit_supression_2/Y{y_index + 1}_sfs1_it{i}"
	os.makedirs(folder, exist_ok = True)

	for feature_used in features_used:
		if feature_used in [2061, 2062]:
			continue
		parameters.remove(feature_used)

	# building the parameters set
	
#	zipped_parameters = zip([features_used] * len(parameters), [weights_used] * len(parameters), parameters)
	zipped_parameters = build_zip_parameters(features_used, weights_used, parameters, folder)
#	print([features_used] * len(parameters), [weights_used] * len(parameters), parameters)
	# Number of processes to run in parallel
	num_processes = 4

	# sequential evaluation
#	for features_to_use, weights_to_use, quad_to_scan in zipped_parameters:
#		TrainModel(features_to_use, weights_to_use, quad_to_scan)

	with Pool(num_processes) as pool:
		pool.starmap(TrainModel, zipped_parameters)

	# gathering the data after the training is over
	train_summary = gather_data2(folder)

	train_summary_sorted = train_summary.sort_values(by = ['total_loss'])

	optimal_features = train_summary_sorted['features_ids'].values[0]
	optimal_weights = list(train_summary_sorted['weights'].values[0].ravel())
	optimal_score = train_summary_sorted['features_ids'].values[0]

	print(optimal_features)
	print(optimal_weights)
	print(optimal_score)
