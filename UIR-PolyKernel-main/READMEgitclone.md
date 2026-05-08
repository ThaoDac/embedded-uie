# UIR-PolyKernel

[Underwater Image Restoration via Polymorphic Large Kernel CNNs](https://arxiv.org/abs/2412.18459)
<div>
<span class="author-block">
  Xiaojiao Guo<sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    Yihang Dong<sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    <a href='https://cxh.netlify.app/'>Xuhang Chen</a>
  </span>,
  <span class="author-block">
    Weiwen Chen
  </span>,
  <span class="author-block">
    Zimeng Li<sup> 📮</sup>
  </span>,
  <span class="author-block">
    <a href='https://lzeeorno.github.io/'>FuChen Zheng</a>
  </span>,
  <span class="author-block">
    <a href='https://cmpun.github.io/'>Chi-Man Pun</a><sup> 📮</sup>
  </span>
  ( 👨‍💻‍ Equal contributions, 📮 Corresponding author)
</div>

<b>University of Macau, SIAT CAS, Huizhou Univeristy, Shenzhen Polytechnic University, The Hong Kong University of Science and Technology (Guangzhou), Baoshan Univeristy</b>

In <b>_IEEE International Conference on Acoustics, Speech, and Signal Processing 2025 (ICASSP 2025)_</b>

# ⚙️ Usage

## Training
You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

## Inference

Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.

```bash
python test.py
```

# Citation

```bib
@inproceedings{guo2025underwater,
  title={Underwater Image Restoration via Polymorphic Large Kernel CNNs},
  author={Guo, Xiaojiao and Dong, Yihang and Chen, Xuhang and Chen, Weiwen and Li, Zimeng and Zheng, FuChen and Pun, Chi-Man},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
accelerate config
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine                                                                                                                                                             
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                                                     
multi-CPU                                                                                                                                                                
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                                                                               
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: NO                                        
Do you want to use Intel PyTorch Extension (IPEX) to speed up training on CPU/XPU? [yes/NO]:                                                                             
Do you want accelerate to launch mpirun? [yes/NO]: NO                                                                                                                    
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO                                                                                                        
How many processes should be used for distributed training? [1]:1                                                                                                        
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?                                                                                                                                      
no                                                                                                                                                                       
accelerate configuration saved at /home/ndpthao/.cache/huggingface/accelerate/default_config.yaml   