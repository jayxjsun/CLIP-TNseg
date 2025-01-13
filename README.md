# CLIP-TNseg: A Multi-Modal Hybrid Framework for Thyroid Nodule Segmentation in Ultrasound Images
This repository contains the code used in the paper ["CLIP-TNseg: A Multi-Modal Hybrid Framework for Thyroid Nodule Segmentation in Ultrasound Images"](https://arxiv.org/abs/2412.05530)



### Dependencies
This code base depends on pytorch, torchvision and clip (`pip install git+https://github.com/openai/CLIP.git`).


### Datasets

* `DDTI`: from “An open access thyroid ultrasound image database”
* `PKTN`: our newly collected dataset, available at https://drive.google.com/file/d/12sIrfY7GMeCnzY0GsamU5FnqkU-qVjQQ/view?usp=drive_link
* `TN3K`: https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation
* `data sourced from Internet`: https://aistudio.baidu.com/datasetdetail/289158
* `comprehensive dataset we constructed`: https://drive.google.com/file/d/1N1AOMjID9NyZlRs_IvjhRb9C_iMS2E95/view?usp=sharing

### Third Party Dependencies
Third party dependencies are required. Run the following commands in the `third_party` folder. 
```bash
git clone https://github.com/ChenyunWu/PhraseCutDataset.git
```
Replace the `data` folder with comprehensive dataset we constructed and rename `PhraseCutDataset` to `thyroid`.


### Weights

The MIT license does not apply to our weights. 

We provide our model weights here.

https://drive.google.com/file/d/1SW0sZ_3alqtEaxU2_5Bfe2KFZa9QzylJ/view?usp=drive_link


### License

The source code files in this repository (excluding model weights) are released under MIT license.


### Citation
```
@misc{sun2024cliptnsegmultimodalhybridframework,
      title={CLIP-TNseg: A Multi-Modal Hybrid Framework for Thyroid Nodule Segmentation in Ultrasound Images}, 
      author={Xinjie Sun and Boxiong Wei and Yalong Jiang and Liquan Mao and Qi Zhao},
      year={2024},
      eprint={2412.05530},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05530}, 
}

```
