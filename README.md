#### Dependencies：

Code has been tested on:

- Python 3.7 
- OS: Windows 10 
- GPU: RTX3060
- CUDA: 11.0
- Tensorflow_gpu: 2.4.0
- Keras: 2.4.3

#### Pip

Use the following command to establish the environment:

```
pip install -r requirements.txt
```



### Quick start

#### Datasets

We use two famous datasets: Stanford Sentiment Treebank (SST) and Large Movie Review Dataset (IMDB). They have been downloaded and saved in the SST-2 and IMDB folders.

#### Models

Our experiments mainly use four deep learning networks: BiLSTM, LSTM, CNN, and CNN-LSTM. For each network and dataset, we will train 10 models that meet the experimental requirements.

Use the following commands to train your own models:

- Download Jupyter Notebook or Pycharm Professional
- choose the network and dataset you want, and find the model code you want to train in folder **Src4Model**

Here is an example:

If you want to train your model on network BiLSTM using SST dataset, you can open folder **Src4Model** and choose BiLSTM_SST.ipynb to run. 



### Workflow

#### Train models

Create Model folder, and follow the instructions of Quick Start/Models to train all the models in this experiment (or the ones you need).

#### Calculate Input metrics and Output metrics

Create Metric folder.

Use the following commands to calculate input and output metrics:

- Download Jupyter Notebook or Pycharm Professional

- For SST dataset, run ExtractMetrics4SST.ipynb

  For IMDB dataset, run ExtractMetrics4IMDB.ipynb

- You can change the network model you want to calculate by changing the *'model_path'* in the code.

#### Calculate Coverage metrics

Create Coverage folder.

Use the following commands to calculate coverage metrics:

- For SST dataset, run:

```
python SST_coverage.py
```

- For IMDB dataset, run:

```
python cov_test_IMDB.py
```

You can change the network model you want to calculate by changing the *'netname'* in the code.

Due to the fact that 10 models are trained between each network and each dataset in our experiment, we use the *'T'* variable in the code to control the currently calculated model (the value of the *'T'* variable is 0-9) when calculating coverage metrics. If you only used one model in your experiment, you can ignore the *'T'* variable in the code.

#### Code for RQ1

Create PCA folder.

Use the following commands to get PCA results:

```
python pca.py
```

#### Codes for RQ2 and Discussion in 5.2

In RQ2, we use 6 deep learning classifiers, including 3 supervised classifiers and 3 unsupervised classifiers. After calculating all indicator values, you can use the following code to reproduce RQ2 in our paper:

- For supervised classifiers:

```
python RQ2 codes/supervised.py
```

You can change the network you want by changing the *'netname'* in the code.

You can change the dataset you want by changing the *'dataname'* in the code.

You can change the model you want by changing the *'i'*  variable in the code (the value of the *'i'* variable is 0-9).

You can change the classifier you want by changing the functions in code.

Here are some examples:

If you want to get the result on classifier LR using up-sampling, then change all the *'upper_sampling_DecisionTree'* to *'upper_sampling_lr'*.

If you want to get the result on classifier GaussianNB using under-sampling, then change all the '*upper_sampling_DecisionTree*' to *'under_sampling_GaussianNB'*.

- For unsupervised classifiers:

  - For classifier KMeans and MiniBatch KMeans:

  ```
  python K-means.py
  ```

  You can change the network, dataset, models and classifiers by the same method as above.

  - For classifier Agglomerative:

  ```
  python Agglomerative.py
  ```

  You can use down-sampling by modifying the method name in line 347, 362, 390, 406 to '*under_sampling*'

#### Code for RQ3

Use the following commands to calculate Information Gain:

```
python INF Gain/information gain.py
```

#### Code for Discussion in 5.1

Use the following commands to calculate the performance of single metric:

```
python calcu_Deepgini.py
```

You can change the network you want to calculate by changing the *'netname'* in the code.

You can change the dataset you want to calculate by changing the *'dataname'* in the code.

#### Code for Discussion in 5.3

Use the following commands to calculate the performance of metrics in cross projects:

```
python CrossProject.py
```

You can change the network you want to calculate by changing the *'pre_net'* and *'new_net'* in the code.

You can change the dataset you want to calculate by changing the *'data'* in the code.

