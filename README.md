# Identification of "AIA terminal nodule" and "AIA central nodule" using DenseNetFCN-3D.

Based on two different 3D movies (`20190805_SJR3.2.2_w1_s2.nd2` and `20190805_322_w1_s2.nd2`), provided by the [Laboratory of the Physics of Biological Systems](https://www.epfl.ch/labs/lpbs/), the task is to identify and distinguish two parts (the *terminal nodule* and the *central nodule*) of a neuron type called AIA.  
3D frames of videos were used as isolated images for training, using a DenseNetFCN-3D with 4 dense blocks of 3 layers per block.  


### Deliverables:
- `DataGenerator.py` python file: Create generator for training and validation, user may apply image augmentation on the fly.  
- `DenseNet3D.py` python file: Implementation of the [DenseNetFCN-3D from GalDude33](https://github.com/GalDude33/DenseNetFCN-3D).   
- `generateFramesMasksFromVideo.ipynb` notebook: Generate frames/masks with desired shape from the video/ground_truth files.  
- `Training.ipynb` notebook: Feed the DenseNetFCN-3D with frames/masks pairs.   
- `VisualizationPrediction.ipynb` notebook: Predict an example mask with the DenseNetFCN-3D model.  
- `ConfusionMatrix.ipynb` notebook: Compute the accuracy and the F1 score for the 'AIA terminal nodule' (class 1) , the 'AIA central nodule'(class 2) and the other cells (class 0).   

(keep all the files in the same directory)

### How to train the DenseNetFCN-3D, predict masks and note the model:
We executed the code on Google Colab by following these steps:
1. The training samples and their corresponding masks need to be stored as `.npy` files. Their names should be `frame_i.npy`and `mask_i.npy` respectively (for i going from 0 → maxTime) and frames/masks need to be stored in two different folders. If you gives the good PATHs in the second cell of `generateFramesMasksFromVideo.ipynb`, this notebook does this steps for you.  
P.S: To help our model, we labeled *ALL* cell parts that were not AIA central/terminal noduls with the same value. Thus, the model only has to distinguish 3 types of cells.
2. Give the wanted paths and variables values to all the second cell's variables of `Training.ipynb` notebook. Then execute all cells. It will train a DenseNetFCN-3D and save the model.
3. Once the model has been trained, it is possible to have a visual idea of the prediction with the `VisualizationPrediction.ipynb` notebook. First, change the variables/paths of the second cell and then run all the cells. The last cell shows the image, the mask and finally the predicted mask.
4. Finally, the `ConfusionMatrix.ipynb` notebook can be used (once the PATHs from cell 2 have been changed) to compute the accuracy and the F1 score for the 'AIA terminal nodule' (class 1) , the 'AIA central nodule'(class 2) and the other cells (class 0).

### REFERENCES: [DenseNet3D from GalDude33](https://github.com/GalDude33/DenseNetFCN-3D)
#### Collaborators: Victor Stimpfling, Loris Pilotto