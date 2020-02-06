#  written by kylesim
from argparse import ArgumentParser
from sklearn.datasets.samples_generator import make_blobs

def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-s', '--n_samples', type=int,
	default=400, help=''
	)
	parser.add_argument(
	'-c', '--n_centers', type=int,
	default=4, help=''
	)
	parser.add_argument(
	'-f', '--n_features', type=int,
	default=2, help=''
	)
	parser.add_argument(
	'-d', '--std', type=float,
	default=0.9, help=''
	)
	return parser.parse_args()
PARAM = get_param()


def gen_data():
	if PARAM.n_centers == 0:
		sample_centers = [[1, 1], [-2, -1], [1, -2], [1, 8]]
		X, y = make_blobs(n_samples=PARAM.n_samples, n_features=PARAM.n_features, centers=sample_centers, cluster_std=PARAM.std)
	else:
		X, y = make_blobs(n_samples=PARAM.n_samples, n_features=PARAM.n_features, centers=PARAM.n_centers, cluster_std=PARAM.std)

	#print(X.shape)
	#print(y.shape)
	zipped = zip(X, y)
	for xy in zipped:
		print("%s\t%s" % ("\t".join(map(str, xy[0])), xy[1]))


if __name__ == '__main__':
	gen_data()
