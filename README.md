# EMcopilot: Generate realistic electron microscopy images for label-free nanocatalyst segmentation

<p align="center">
<a href="https://doi.org/10.1038/s41524-025-xxxxx-x"><img src="https://img.shields.io/badge/Paper-npj%20Computational%20Materials-black.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAS1BMVEVHcEwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD///+Ojo7n5+fT09MxMTG6urqfn59PT0/09PRra2shISF+fn46JhgGAAAADHRSTlMA0hh4smIK+ZDlSjcbp4U+AAAB5UlEQVRYhaVX7RaCIAxVsQQFzc96/yctLRiMaXB2fxnt3sPG2EZRnKBqRV2qRutGlbVoqzM7Gp1UGkHJLpV9FxH7pyHuKXTZ0PQdjfwrcbugHxK3S3pVXtN3lBfxvP2n7zjdhEzjay1pfp3K17pm8kmFLD6hkOy/BYpDYvx9BGdR5fO19vOBzJ+n+1i2bXlG/5eXDozzw3y/5sEcGOZTJ+44/7dp3Sn7Z28A6xKaNfZmBSew9A9L+PyajI8V+SGpDXgUzDdmIrcgwtUVBF4GA21BHAKo/oDXnpbFFtqqnd+hCI7Oejbr9FqWbQYBfBIdlcSDM+9/K0+3FRSEI4xRBZ0w33PrgWwVlcVzLLCcCXzyuY0E+lhgPBVo8SHmCgiikGQJ1MRFzBIo40PIE1BF3ImyBJoi4ucJaL4A2wV2ENnHyE4kdiqzL1N8nbMEKqKgEPXgVEBRJc0JQAF0BWVAtpIoqlDSCIEV2XZEWQcBqKCbXTKxB1FjgaoMAtBhws4iyN7qajhEDDpD0F9tdw3DCI0FHIYe+fJN7ZgTbgH8NWPkVdBZXHsPB4x5cHCdEJb8RPCmpIQROYY34vCHLPaYxx80+aMuf9jmj/v8Bwf/ycN/dBXsZ1/Bf3gW7KfvAd7j+4vU5/8bY+ib2he8lwIAAAAASUVORK5CYII=" alt="npj Computational Materials"></a>
<a href="https://doi.org/10.1093/mam/ozae044.211"><img src="https://img.shields.io/badge/Poster-M&M2024-2c90bb.svg" alt="M&M2024"></a>
<a href="https://doi.org/10.5281/zenodo.14994375"><img src="https://img.shields.io/badge/Datasets-Zenodo-0067c9.svg" alt="Zenodo Datasets"></a>
<a href="https://huggingface.co/wy337/EMcopilot"><img src="https://img.shields.io/badge/Models-HuggingFace-yellow.svg?logo=huggingface" alt="Hugging Face Models"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
<!-- 
<a href="https://doi.org/10.1038/s41524-025-xxxxx-x"><img src="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/head.png" alt="Title figure" width="800"></a>
 -->
</p>

<p align="center">
<img src="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/toc.png" alt="TOC figure" width="800"></a>
</p>

| **Category**       | **Scope of EMcopilot** |
|:--------------------:|:--------------:|
| Imagings | TEM, STEM, BF, DF, HAADF |
| Applications | Particle size distribution, single-atom catalysts detection, semantic segmentation |
| Materials | PtSn-NPs@Al₂O₃, Au-NPs@ZSM-5, Pd-NPs@C，PtSn-Cluster@Al₂O₃, Pt-SACs@C, Pt-SACs@NC, Pd-SACs@NC, Ni-SACs@NC, and Ru-SACs@NC |

</details>
</div>


## 🎥 Demonstration Movie - Image Generation Process
<p align="center">
<video src="https://github.com/user-attachments/assets/6080d6be-b48a-4388-94e3-568ed093c813" 
       autoplay 
       loop 
       muted 
       style="max-width:100%; height:auto;">
  Your browser does not support the video tag.
</video>
</p>


- *This movie* illustrates the generation and iterative refinement of synthetic EM images for various supported catalysts as training epochs advance, progressively assimilating real image features.  
- *Toward the end*, a quick-flash sequence visually compares the final synthetic images with their corresponding experimental counterparts.

##  What Can These Generated Images Do?

### 🔬 For Microscopists
- Automated analysis of TEM images (even SEM, SPM, and OM, etc), without multislice simulations or manual labeling.

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
  <img src="https://github.com/WenhaoYuan337/EMcopilot/blob/main/resources/demo4.png" alt="Catalysis Use Case3" width="700">
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
- `Zenodo`: [https://doi.org/10.5281/zenodo.14994375](https://doi.org/10.5281/zenodo.14994375)

Models are also available on：
-  `Hugging Face`: [https://huggingface.co/wy337/EMcopilot](https://huggingface.co/wy337/EMcopilot)

  
## Citation
If you find our code or data useful in your research, please cite:
```
Yuan W., Yao B., Tan S. et al. Generative learning of morphological and contrast heterogeneities for self-supervised electron micrograph segmentation. npj Comput Mater 11, xxx (2025). https://doi.org/10.1038/s41524-025-xxxxx-x
```

[Download citation ⬇](https://citation-needed.springer.com/v2/references/10.1038/s41524-025-xxxxx-x?format=refman&flavour=citation)



## Contact
For any questions regarding EMcopilot, including the paper or implementation, please feel free to contact me at <wy337@cornell.edu>.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
This work is the result of collaboration between:
- [Qian's lab, National University of Singapore (Prof. Qian He)](https://heqian.org)
- [PEESE lab, Cornell University (Prof. Fengqi You)](https://www.peese.org/)


