# M2 ISI - Advanced Machine Learnig

## Chest X-Ray images for COVID 19's prediction


## Repository


__Notebook__ : 
  + _Data_generation.ipynb_: qui prétraite les images et fournis les X, Y utiles à l'entrainement du réseau
  + _Data_exploration.ipynb_: qui examine et affiche les données utilisées.
  + _Resnet_final.ipynb_: application du réseau Resnet, études des performances et visualisation des critères de décision du réseau.
  + _VGG16.ipynb_: application du réseau VGG16 et étude des performances de celui-ci.
  + _CNN.ipynb_: tentative sur un réseau CNN "from scratch".
  + _segmentation.py_: code qui segmente les x-rays covid fournis de la base github en utilisant les masques donnés.
  
  
## Prerequisite

  + Python 3.6/3.7
  + Jupyer (pip install jupyterlab)
  + Tensorflow 2.4.0 
  + Keras 2.4.3
  
## Reference

Cohen, Morrison and Lan Dao, image data collection, [Github](https://github.com/ieee8023/covid-chestxray-dataset)
