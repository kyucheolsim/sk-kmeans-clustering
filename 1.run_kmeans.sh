# written by kylesim

N_CLUSTERS=4
MAX_ITER=500

BIN_TRAIN="train_kmeans.py"
BIN_PREDICT="predict_kmeans.py"
DATA_PATH="./data/sample_data_kmeans.txt"
KMS_MODEL_PATH="./model/model_kmeans.bin"
MBK_MODEL_PATH="./model/model_mb_kmeans.bin"

run_train()
{
	if [ "$1" == "ssd" ]; then
		python ${BIN_TRAIN} --action=ssd --data_path=${DATA_PATH} --labeled
	
	elif [ "$1" == "kmeans" ]; then
		python ${BIN_TRAIN} --action=kmeans --n_clusters=${N_CLUSTERS} --max_iter=${MAX_ITER} --data_path=${DATA_PATH} --model_path=${KMS_MODEL_PATH} --labeled --debug
	elif [ "$1" == "mb_kmeans" ]; then
		python ${BIN_TRAIN} --action=mb_kmeans --n_clusters=${N_CLUSTERS} --max_iter=${MAX_ITER} --data_path=${DATA_PATH} --model_path=${MBK_MODEL_PATH} --labeled --debug
	fi
}

run_predict()
{
	if [ "$1" == "batch" ]; then
		python ${BIN_PREDICT} --action=batch --data_path=${DATA_PATH} --model_path=${KMS_MODEL_PATH} --labeled
	elif [ "$1" == "accuracy" ]; then
		python ${BIN_PREDICT} --action=accuracy --data_path=${DATA_PATH} --model_path=${KMS_MODEL_PATH} --labeled --debug
	fi
}

case $1 in
	'train')
		shift
		run_train $1
	;;
	'predict')
		shift
		run_predict $1
	;;
esac
