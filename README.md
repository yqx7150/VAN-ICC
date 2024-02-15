# VAN-ICC   
Variable augmentation network for invertible MR coil compression   
X Liao, B Huang, S Wang, D Liang, Q Liu    
Magnetic Resonance Imaging, Volume 108, May 2024, Pages 116-128         
https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225   

To improve the efficiency of multi-coil data compression and recover the compressed image reversibly, increasing the possibility of applying the proposed method to medical scenarios. A deep learning algorithm is employed for MR coil compression in the presented work. The approach introduces a variable augmenta-tion network for invertible coil compression (VAN-ICC). This network utilizes the inherent reversibility of normalizing flow-based models. The aim is to enhance the readability of the sentence and clearly convey the key components of the algorithm. By applying the variable augmentation technology to image/k-space variables from multi-coils, VAN-ICC trains the invertible network by finding an invertible and bijective function, which can map the original data to the compressed counterpart and vice versa. Experiments conducted on both fully-sampled and under-sampled data verified the effectiveness and flexibility of VAN-ICC. Quantitative and qualitative comparisons with traditional non-deep learning-based approaches demonstrated that VAN-ICC carries much higher compression effects. The proposed method trains the invertible network by finding an invertible and bijective function, which improves the defects of traditional coil compression method by utilizing inherent reversibility of normalizing flow-based models. In addition, the application of variable augmentation technology ensures the implementation of reversible networks. In short, VAN-ICC offered a competitive advantage over other traditional coil compression algorithms.   
       
        
## Graphical representation
 <div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig6.png" width = "400" height = "450">  </div>
 
Performance exhibition of “multi-view noise” strategy. (a) Training sliced score matching (SSM) loss and validation loss for each iteration. (b) Image quality comparison on the brain dataset at 15% radial sampling: Reconstruction images, error maps (Red) and zoom-in results (Green).

 <div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>

Pipeline of sampling from the high-dimensional noisy data distribution with multi-view noise and intermediate samples. (a) Conceptual dia-gram of the sampling on high-dimensional noisy data distribution with multi-view noise. (b) Intermediate samples of annealed Langevin dynamics.


## Reconstruction Results by Various Methods at 85% 2D Random Undersampling.
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig11.png"> </div>

Reconstruction comparison on pseudo radial sampling at acceleration factor 6.7 . Top: Reference, reconstruction by DLMRI, PANO, FDLCP; Bottom: Reconstruction by NLR-CS, DC-CNN, EDAEPRec, HGGDPRec. Green and red boxes illustrate the zoom in results and error maps, respectively.

    
