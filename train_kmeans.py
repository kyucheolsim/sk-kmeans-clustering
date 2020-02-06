# written by kylesim
from argparse import ArgumentParser
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import accuracy_score
from util_kmeans import *

# For large scale learning (say n_samples > 10k) MiniBatchKMeans is probably much faster than the default batch implementation.

def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-a', '--action', type=str,
	default='', help='( ssd | kmeans | mb_kmeans )'
	)
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='', help='input data path'
	)
	parser.add_argument(
	'-b', '--batch_size', type=int,
	default=100, help='batch size for mb_kmeans'
	)
	parser.add_argument(
	'-c', '--n_clusters', type=int,
	default=2, help='number of clusters for kmeans and mb_kmeans'
	)
	parser.add_argument(
	'-i', '--max_iter', type=int,
	default=300, help='max number of iterations for kmeans and mb_kmeans'
	)
	parser.add_argument(
	'-k', '--max_k', type=int,
	default=10, help='max number of clusters to find best k'
	)
	parser.add_argument(
	'-m', '--model_path', type=str,
	default='', help='path to save trained model'
	)
	parser.add_argument(
	'--labeled', action='store_true',
	default=False, help='labeled data? (x ... y)'
	)
	parser.add_argument(
	'--debug', action='store_true',
	default=False, help='print input parameters?'
	)
	return parser.parse_args()
PARAM = get_param()
if PARAM.debug:
	print("# "+str(PARAM))


def print_k_ssd(X, max_k = 10, max_iter = 300):
	print("k\tssd")
	for k in range(2, max_k+1, 1):
		kmeans = KMeans(init = 'k-means++', n_clusters = k, max_iter = max_iter, tol = 0.0001, n_init = 10, random_state = 0, verbose = 0)
		kmeans = kmeans.fit(X)
		#print(kmeans.labels_) # (n_samples,)
		#print(kmeans.cluster_centers_) # (n_clusters, n_features)
		print("{}\t{}".format(k, kmeans.inertia_)) # sum of squared distance


if __name__ == '__main__':
	X, y = load_data(PARAM.data_path, PARAM.labeled)

	if PARAM.action == 'ssd':
		print_k_ssd(X, PARAM.max_k, PARAM.max_iter)
	else:
		if PARAM.action == 'kmeans':
			model = KMeans(init = 'k-means++', n_clusters = PARAM.n_clusters, max_iter = PARAM.max_iter, tol = 0.0001, n_init = 10, random_state = None, verbose = 0)
			model.fit(X)
		elif PARAM.action == 'mb_kmeans':
			model = MiniBatchKMeans(init = 'k-means++', n_clusters = PARAM.n_clusters, max_iter = PARAM.max_iter, batch_size = PARAM.batch_size, tol = 0.0, n_init = 10, max_no_improvement = 10, random_state = None, verbose = 0)
			model.fit(X)
		else:
			raise ValueError('unknown action')

		if y:
			# labeled data, run some evaluations
			ct_map = get_ct_map(y, model.labels_, PARAM.n_clusters)
			pred_labels = conv_cluster_labels(model.labels_, ct_map)
			accuracy = accuracy_score(y, pred_labels)
		else:
			ct_map = None
			accuracy = 0.0

		model_map = {}
		model_map['accuracy'] = accuracy
		model_map['ct_map'] = ct_map
		if PARAM.debug:
			print_model_map(model_map)
		if PARAM.model_path:
			# save the model for the future prediction
			model_map['model'] = model
			save_model_map(model_map, PARAM.model_path)
