# 02456-deep-learning-project12
This repository contains the code and paper for project 12 in the 02456 deep learning course at DTU. The authors are Rasmus Arpe Fogh Jensen (s134843) and Thomas Pethick (s144448). 

First stepping stone of the project was to implement and obtain similar results as the article *Piczak: Environmental sound classification with convolutional neural networks* on the urbanSound-8K dataset. The implemntation and results are displayed in notebook 'Step1_piczak-urbansound-8K.ipynb', but ommitted from the paper, since they seemed irrelevant for the final paper. Considerable amount of work was however also put into this, which is why they are included in the repository. 

The second step was to implement a semi-supervised learning approach for environmental sound classification utilizing unlabeled data. Specifically, the results were obtained using the datasets ESC-US (250.000 unlabeled sounds for pretraining) and ESC-50 (2000 labeled sounds, 50 different classes). A display of code is shown in the jypyter notebook'Step2_ESC-semi-supervised-learning.ipynb'. Note that the data-processing is ommitted from the notebook and instead implemented as its own class in DataHandler.py. The DataHandler class allows dynamically loading batches into memory used for training. 

*Note:* To run the jupyter notebooks, you will need to include a folder '/datasets/' and download the relevant datasets into the folder. The datasets are found at these locations: [UrbandSound-8K](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html), [ESC-50](https://github.com/karoldvl/ESC-50) and [ESC-US](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT).

The end goal was a paper with the proposed method and results. The paper can be found in `paper/semi-supervised-learning.pdf`. The abstract is included below. 

## Abstract of paper
### Title: Environmental sound classification with semi-supervised learning
In this paper, we explore a semi-supervised learning approach for environmental sound classification. A convolutional autoencoder is used for pre-training the weights in the network. Two different methods for the invertions of the max-pooling layers are examined in the decoder; upsampling and unpooling. The semi-supervised approach is benchmarked against a supervised approach with similar architecture on a public available dataset. The results show that semi-supervised learning yields slightly better performance utilizing unlabeled data.
