# written by kylesim

BIN="gen_data_kmeans.py"
DATA_DIR="./data"

source ./venv/bin/activate
mkdir -p ${DATA_DIR}

run_sample()
{
	echo "generate sample data..."
	python ${BIN} --n_samples=1000 --n_centers=0 --n_features=2 > ${DATA_DIR}/sample_data_kmeans.txt
}

run_small()
{
	echo "generate small data..."
	python ${BIN} --n_samples=1000 --n_centers=4 --n_features=2 > ${DATA_DIR}/small_data_kmeans.txt
}

run_medium()
{
	echo "generate medium data..."
	python ${BIN} --n_samples=10000 --n_centers=5 --n_features=2 > ${DATA_DIR}/medium_data_kmeans.txt
}


run_large()
{
	echo "generate large data..."
	python ${BIN} --n_samples=100000 --n_centers=10 --n_features=2 > ${DATA_DIR}/large_data_kmeans.txt
}

case $1 in
	'sample')
		run_sample
	;;
	'small')
		run_small
	;;
	'medium')
		run_medium
	;;
	'large')
		run_large
	;;
esac
