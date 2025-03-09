# EMcopilot: Your label-free copilot for automated electron microscopy image analysis

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

## Installation

Install required packages using:
```bash
pip install -r requirements.txt
```
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
Copyright | 2025 Qian's Lab@NUS 

