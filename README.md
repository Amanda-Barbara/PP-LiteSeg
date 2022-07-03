# PP-LiteSeg
Pytorch implement of PP-LiteSeg Semantic Segmentation.

## Requirement

- OpenCV 4.1
- Python 3.8   
- Pytorch 1.8

## Model

The architecture overview. PP-LiteSeg consists of three modules: encoder, aggregation and decoder. A lightweight network is
used as encoder to extract the features from different levels. The Simple Pyramid Pooling Module (SPPM) is responsible for aggregating
the global context. The Flexible and Lightweight Decoder (FLD) fuses the detail and semantic features from high level to low level and
outputs the result. Remarkably, FLD uses the Unified Attention Fusion Module (UAFM) to strengthen feature representations.

![PP-LiteSeg](/image/net.png)

## Train

## Test

## Reference


https://github.com/PaddlePaddle/PaddleSeg

```
@article{
  title={PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model},  
  author={Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu, Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang, Baohua Lai, Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma},
  journal={arXiv preprint arXiv:2204.02681},
  year={2022}
}
```
