# EMcopilot: Generate realistic electron microscopy images for label-free nanostructure segmentation

<p align="center">
<a href="https://arxiv.org/abs/2407.19544"><img src="https://img.shields.io/badge/arXiv-2407.19544-b31b1b.svg?logo=arxiv&logoColor=red" alt="arXiv"></a>
<a href="https://doi.org/10.5281/zenodo.14994375"><img src="https://img.shields.io/badge/Datasets-Zenodo-blue.svg?logo=databricks" alt="Zenodo Datasets"></a>
<a href="https://huggingface.co/wy337/EMcopilot"><img src="https://img.shields.io/badge/Models-HuggingFace-yellow.svg?logo=huggingface" alt="Hugging Face Models"></a>
<a href="https://doi.org/10.1093/mam/ozae044.211"><img src="https://img.shields.io/badge/Poster-M&M2024-blue.svg" alt="M&M2024"></a>
<a href="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/Supplementary%20Information.pdf"><img src="https://img.shields.io/badge/SI-PDF-lightgrey.svg" alt="Supplementary Information PDF"></a>
<a href="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/Supplementary_Movie_1.mp4"><img src="https://img.shields.io/badge/SI-Movie1-lightgrey.svg" alt="Supplementary Movie 1"></a>
<a href="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/Supplementary_Movie_2.mp4"><img src="https://img.shields.io/badge/SI-Movie2-lightgrey.svg" alt="Supplementary Movie 2"></a>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2407.19544"><img src="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/toc.png" alt="TOC figure" width="1000"></a>
</p>

<div align="center">
<details>
<summary><b>Scope of EMcopilot</b> (click to expand)</summary>

| **Category**       | **Coverage** |
|--------------------|--------------|
| **Imaging Modes**  | TEM, STEM, BF, DF, HAADF |
| **Demo Materials** | PtSn-NPs@Al₂O₃, Au-NPs@ZSM-5, Pd-NPs@C，PtSn-Cluster@Al₂O₃, Pt-SACs@C, Pt-SACs@NC, Pd-SACs@NC, Ni-SACs@NC, and Ru-SACs@NC |
| **Applications**   | Semantic segmentation, Particle size distribution, Single-atom/clusters detection |

</details>
</div>


## 🎥 Demonstration Movie - Image Generation
- *This movie* illustrates the generation and iterative refinement of synthetic EM images for various supported catalysts as training epochs advance.  
- *It captures* the evolution of the generator’s learning process, progressively assimilating real image features.  
- *Toward the end*, a quick-flash sequence visually compares the final synthetic images with their corresponding experimental counterparts.

<p align="center">
<video src="https://github.com/user-attachments/assets/6080d6be-b48a-4388-94e3-568ed093c813" 
       autoplay 
       loop 
       muted 
       style="max-width:100%; height:auto;">
  Your browser does not support the video tag.
</video>
</p>


##  What Can These Generated Images Do?

### 🔬 For Microscopists
- Automated analysis of TEM images (even SEM, SPM, and OM, etc), without any manual labeling.

<p align="center">
  <img src="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/demo1.png" alt="Microscopy Use Case" width="700">
</p>

---

### ⚛️ For Catalysis Researchers
- Precise and high-throughput extraction of morphological descriptors for bulk/supported catalysts, still, without any manual labeling.
<p align="center">
  <img src="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/demo2.png" alt="Catalysis Use Case1" width="700">
</p>

- Besides sizing, shape also matters. Obtain shape statistics such as eccentricity and circularity from thousands of particles within seconds.
<p align="center">
  <img src="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/demo3.png" alt="Catalysis Use Case2" width="700">
</p>

- Deeper insights into the spatial distributions and organizations among active sites.
<p align="center">
  <img src="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/xxx.png" alt="Catalysis Use Case3" width="700">
</p>


## Installation

Install required packages using:
```bash
pip install -r requirements.txt
```


## Usage

1. **Train and predict segmentation models**:
     * `00_01_sam_binary_masking.py` -  Generates coarse masks using the SAM model.
3. **Generate and analyze synthetic masks**:
     * `02_01_sam_mask_analysis.py` - Analyzes SAM mask properties and extracts morphology prior.
     * `02_02_random_mask_generate.py` - Generates synthetic masks by augmenting existing masks.
5. **Generate and evaluate images**:
     * `03_01_p2p_train.py` - Trains a Pix2Pix model for mask-to-EMimage translation.
     * `04_01_pix2pix_predict.py` - Runs inference using the trained Pix2Pix model.
6. **Domain Adaptation**:
     * `05_01_domain_adaptation.py` - Applies domain adaptation, including noise and contrast augmentation.
7. **UNet++ Training and Inference**:
     * `06_01_unet++_train.py` - Trains a CBAM-enhanced UNet++ model for segmentation.
     * `07_01_unet++_predict.py` - Performs inference using the trained UNet++ model.
8. **Analyze DM4 microscopy data**:
     * `08_01_in_situ_analysis.py` - Analyze HAADF-STEM images of supported nanoparticles in real time.


## Datasets 
All data that support the findings of this study, including experimental data, synthetic data and model checkpoints, are available on :
- `Zenodo`: Yuan, W., Yao, B., Tan, S., You, F., & He, Q. (2025). Source Data for: Generative learning of morphological and contrast heterogeneities for self-supervised electron micrograph segmentation [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.14994375](https://doi.org/10.5281/zenodo.14994375)

Models are also available on：
-  `Hugging Face`: [https://huggingface.co/wy337/EMcopilot](https://huggingface.co/wy337/EMcopilot)

  
## Citation
If you find our code or data useful in your research, please cite our paper:
```
@misc{yuan2024deepgenerativemodelsassistedautomated,
      title={Deep Generative Models-Assisted Automated Labeling for Electron Microscopy Images Segmentation}, 
      author={Wenhao Yuan and Bingqing Yao and Shengdong Tan and Fengqi You and Qian He},
      year={2024},
      eprint={2407.19544},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2407.19544}, 
}
```
· Copyright © 2025 Qian's Lab ([Website](https://heqian.org/)) · Last updated Aug 14, 2025 · 

