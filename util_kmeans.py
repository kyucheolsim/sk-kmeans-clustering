# written by kylesim
import pickle
import numpy as np


def load_data(data_path, labeled):
	# load input data (train, valid, test, predict)
	X = []
	y = []
	with open(data_path, 'r') as data_file:
		for line in data_file:
			line = line.strip()
			tokens = line.split('\t')
			if labeled is True:
				y.append(int(tokens.pop(-1)))
			X.append(map(float, tokens))
	return X, y


def print_ct_map(ct_map):
	n_clusters = len(ct_map)
	for i in range(0, n_clusters, 1):
		# cluster index -> true index
		print("# {} -> {}".format(i, ct_map[i]))


def get_ct_map(true_labels, cluster_labels, n_clusters):
	# map: cluster index -> true index
	ct_map = {}
	true_labels = np.array(true_labels)
	for k in range(0, n_clusters, 1):
		max_index = np.argmax(np.bincount(true_labels[cluster_labels == k]))
		ct_map[k] = max_index # most frequent true index in cluster k
	return ct_map


def conv_cluster_labels(cluster_labels, ct_map):
	# cluster index -> true index
	conv_labels = np.array([ct_map[i] for i in cluster_labels])
	return conv_labels


def get_simple_accuracy(true_labels, pred_labels):
	correct_count = 0.0
	total_count = len(true_labels)
	for i, y in enumerate(true_labels):
		if y == pred_labels[i]:
			correct_count += 1.0
	accuracy = correct_count / total_count
	return accuracy


def get_np_accuracy(true_labels, pred_labels):
	true_labels = np.array(true_labels)
	pred_labels = np.array(pred_labels)
	accuracy = np.mean(true_labels == pred_labels)
	return accuracy


def save_model_map(model, model_path):
	with open(model_path, 'wb') as model_file:
		pickle.dump(model, model_file)


def load_model_map(model_path):
	with open(model_path, 'rb') as model_file:
		return pickle.load(model_file)

def print_model_map(model_map):
	accuracy = model_map.get('accuracy', None)
	if accuracy:
		print("# Accuracy: {}".format(accuracy))
	ct_map = model_map.get('ct_map', None)
	if ct_map:
		print("# CT Map")
		print_ct_map(ct_map)


