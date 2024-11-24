# VAN-ICC   
Variable augmentation network for invertible MR coil compression   
X Liao, B Huang, S Wang, D Liang, Q Liu    
Magnetic Resonance Imaging, Volume 108, May 2024, Pages 116-128         
https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225   

To improve the efficiency of multi-coil data compression and recover the compressed image reversibly, increasing the possibility of applying the proposed method to medical scenarios. A deep learning algorithm is employed for MR coil compression in the presented work. The approach introduces a variable augmenta-tion network for invertible coil compression (VAN-ICC). This network utilizes the inherent reversibility of normalizing flow-based models. The aim is to enhance the readability of the sentence and clearly convey the key components of the algorithm. By applying the variable augmentation technology to image/k-space variables from multi-coils, VAN-ICC trains the invertible network by finding an invertible and bijective function, which can map the original data to the compressed counterpart and vice versa. Experiments conducted on both fully-sampled and under-sampled data verified the effectiveness and flexibility of VAN-ICC. Quantitative and qualitative comparisons with traditional non-deep learning-based approaches demonstrated that VAN-ICC carries much higher compression effects. The proposed method trains the invertible network by finding an invertible and bijective function, which improves the defects of traditional coil compression method by utilizing inherent reversibility of normalizing flow-based models. In addition, the application of variable augmentation technology ensures the implementation of reversible networks. In short, VAN-ICC offered a competitive advantage over other traditional coil compression algorithms.   
       
        
## The training and testing procedure of VAN-ICC
 <div align="center"><img src="https://github.com/yqx7150/VAN-ICC/blob/main/Fig2.jpg"> </div>
 
Here we take the coil numbers of the input and compressed objects to be 12 and 4 in VAN-ICC-K for example. Average (Ave) denotes the average operator conducted across the channel directions.

## The pipeline of the training process of VAN-ICC
 <div align="center"><img src="https://github.com/yqx7150/VAN-ICC/blob/main/Fig3.jpg"> </div>

Left: The training process of VAN-ICC-I. Right: The training process of VAN-ICC-K. The differences between VAN-ICC-I and VAN-ICC-K are the IFFT and SOS module and that the input and output of VAN-ICC-K are in k-space domain. IFFT stands for the inverse FFT. Average denotes the average operator conducted across the channel directions.


## Compression and reconstruction experiments on image domain
<div align="center"><img src="https://github.com/yqx7150/VAN-ICC/blob/main/Fig4.jpg"> </div>

(a) Comparison of compression results on brain dataset for different methods. (b) Comparison of compression results on cardiac dataset for different methods. Complex-valued PI reconstruction results and the intensity of residual maps is five times magnified. (c) Re-construction results by L1-SPIRiT of the under-sampled cardiac dataset compressed by different methods. Bottom: The 15x absolute difference images between the reference image and reconstruction images.

### Other Related Projects
<div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/algorithm-overview.png" width = "800" height = "500"> </div>
 Some examples of invertible and variable augmented network: IVNAC, VAN-ICC, iVAN and DTS-INN.    
     
  * Variable Augmented Network for Invertible Modality Synthesis and Fusion  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10070774)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iVAN)    
  
  * Variable Augmented Network for Invertible Decolorization (基于辅助变量增强的可逆彩色图像灰度化)  [<font size=5>**[Paper]**</font>](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221205?viewType=HTML)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VA-IDN)    

 * Virtual coil augmentation for MR coil extrapoltion via deep learning  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22001722)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VCA)    

  * Synthetic CT Generation via Invertible Network for All-digital Brain PET Attenuation Correction  [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2310.01885)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)        

  * Invertible and Variable Augmented Network for Pretreatment Patient-Specific Quality Assurance Dose Prediction  [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10278-023-00930-w)       
    
  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)   
