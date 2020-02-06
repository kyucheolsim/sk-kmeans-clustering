# kmeans-clustering
* train kmeans
~~~
usage: train_kmeans.py [-h] [-a ACTION] [-d DATA_PATH] [-b BATCH_SIZE]
                       [-c N_CLUSTERS] [-i MAX_ITER] [-k MAX_K]
                       [-m MODEL_PATH] [--labeled] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  -a ACTION, --action ACTION
                        ( ssd | kmeans | mb_kmeans )
  -d DATA_PATH, --data_path DATA_PATH
                        input data path
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size for mb_kmeans
  -c N_CLUSTERS, --n_clusters N_CLUSTERS
                        number of clusters for kmeans and mb_kmeans
  -i MAX_ITER, --max_iter MAX_ITER
                        max number of iterations for kmeans and mb_kmeans
  -k MAX_K, --max_k MAX_K
                        max number of clusters to find best k
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to save trained model
  --labeled             labeled data? (x ... y)
  --debug               print input parameters?
  ~~~
  * predict kmeans
  ~~~
  usage: predict_kmeans.py [-h] [-a ACTION] [-d DATA_PATH] [-m MODEL_PATH]
                         [--labeled] [--print_y] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  -a ACTION, --action ACTION
                        ( batch | accuracy )
  -d DATA_PATH, --data_path DATA_PATH
                        input data path for prediction
  -m MODEL_PATH, --model_path MODEL_PATH
                        path for trained model
  --labeled             labeled data? (x ... y)
  --print_y             print input true labels? (x ... y pred_y)
  --debug               print model map and input parameters?
  ~~~
