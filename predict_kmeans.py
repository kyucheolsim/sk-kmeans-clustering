# written by kylesim
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from util_kmeans import *

def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-a', '--action', type=str,
	default='', help='( batch | accuracy )'
	)
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='', help='input data path for prediction'
	)
	parser.add_argument(
	'-m', '--model_path', type=str,
	default='', help='path for trained model'
	)
	parser.add_argument(
	'--labeled', action='store_true',
	default=False, help='labeled data? (x ... y)'
	)
	parser.add_argument(
	'--print_y', action='store_true',
	default=False, help='print input true labels? (x ... y pred_y)'
	)
	parser.add_argument(
	'--debug', action='store_true',
	default=False, help='print model map and input parameters?'
	)
	return parser.parse_args()
PARAM = get_param()
if PARAM.debug:
	print("# "+str(PARAM))


def print_result(X, y, pred_labels):
	if y:
		# include true labels
		for i, label in enumerate(pred_labels):
			print("{}\t{}\t{}".format("\t".join(map(str, X[i])), y[i], label))
	else:
		for i, label in enumerate(pred_labels):
			print("{}\t{}".format("\t".join(map(str, X[i])), label))


if __name__ == '__main__':
	if PARAM.model_path:
		model_map = load_model_map(PARAM.model_path)
		if PARAM.debug:
			print_model_map(model_map)
		ct_map = model_map['ct_map']
		model = model_map['model']
	else:
		raise ValueError('model_path not set')

	if PARAM.action == 'batch' or PARAM.action == 'accuracy':
		X, y = load_data(PARAM.data_path, PARAM.labeled)
		pred_labels = model.predict(X)
		if ct_map:
			pred_labels = conv_cluster_labels(pred_labels, ct_map)
		if PARAM.action == 'batch':
			if PARAM.print_y:
				print_result(X, y, pred_labels)
			else:
				print_result(X, None, pred_labels)
		else:
			accuracy = accuracy_score(y, pred_labels)
			print("# Accuracy: {}".format(accuracy))
	else:
		raise ValueError('unknown action')
