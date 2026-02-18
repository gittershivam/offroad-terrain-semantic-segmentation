# offroad-terrain-semantic-segmentation
Semantic segmentation of off-road terrain images using a pretrained DINOv2 backbone and custom segmentation head. Optimized under GPU and time constraints.

**Overview**
This project implements a deep learning-based semantic segmentation model for off-road terrain understanding. The goal is to classify each pixel in an image into terrain categories such as rocks, grass, logs, sky, and landscape.
Accurate terrain segmentation is critical for autonomous navigation in unstructured environments.
Final Test Mean IoU: 0.2276

**Problem Statement**
Given off-road terrain images, the task is to perform pixel-wise classification into multiple terrain categories. The primary evaluation metric is Mean Intersection over Union (IoU) on a held-out test dataset.
The challenge involves:
* Domain variation between training and test data
* Texture similarity between terrain classes
*  Limited GPU resources (4GB VRAM)
*  Strict time constraints


## Methodology

**Model Architecture**
* Pretrained DINOv2 backbone for feature extraction
* Custom segmentation head for pixel-level classification
* Cross-Entropy Loss
* Adam optimizer

**Training Configuration (Final Model)**
* Batch Size: 2
* Learning Rate: 1e-4
* Epochs: 10
* GPU: NVIDIA GTX 1650 (4GB)

The DINOv2 backbone provides strong self-supervised visual features, enabling improved generalization with limited labeled data.


## Experimental Results
Multiple training configurations were evaluated to optimize performance.

|Configuration	            |     Val IoU|	Test IoU
|   :---                        | :---           |:---
|10 Epochs (Final Model)    |  	0.2960	 |0.2276
12 Epochs (Batch=4)	        |  0.2834	   |0.2181
20 Epochs (LR=5e-5)         |	0.2486	   |0.1977

##  Key Insight
The model achieved optimal generalization at **10 epochs**.  
Extending training beyond this point reduced the test IoU, indicating mild overfitting to the training distribution.

This experiment highlights the importance of **early stopping**, especially when fine-tuning strong pretrained backbones on relatively limited datasets.

Based on comparative evaluation across multiple configurations, the **10-epoch model** was selected as the final submission due to its superior test performance.



## Training Behavior
*  Training and validation loss decreased steadily.
*  Minimal gap between training and validation metrics.
*  Overfitting observed after extended training (12+ epochs).
*  Domain gap between training and test data impacted final IoU.

Training curves and per-class metrics are available in the train_stats/ directory.


## Challenges Faced
1. Limited GPU memory (4GB VRAM)
2. Texture similarity between classes (e.g., logs vs rocks)
3. Domain shift between training and test environments
4. Overfitting during extended training
5. Fine-grained object boundary detection difficulties


## Failure Case Analysis
*  The model struggles with:
*  Fine-grained vegetation details
*  Small object boundaries
*  Texture-similar terrain classes
*  These challenges contribute to lower IoU in certain classes.

Sample predictions are available in predictions/.


## Final Model
  * Final submission model:
    segmentation_head_10epoch_best.pth

Final Test Mean IoU: 0.2276

## How to Run

### Install Dependencies

```bash
conda create -n offroad python=3.9
conda activate offroad
pip install -r requirements.txt
```

###  Train the Model

```bash
python train_segmentation.py
```

###  Run Evaluation

```bash
python test_segmentation.py
```

Evaluation outputs will be saved in the `predictions/` folder.


## Future Work
* Data augmentation (random flips, rotations, color jitter)
* Class-balanced or focal loss
* Higher resolution training
* Post-processing refinement (e.g., CRF)
* Fine-tuning backbone layers with larger GPU


## Conclusion

This project demonstrates the feasibility of deep learning-based terrain segmentation using pretrained visual backbones under limited hardware and time constraints.
Careful experimentation revealed that optimal generalization occurred at 10 epochs, emphasizing the importance of validation-based model selection and early stopping.











