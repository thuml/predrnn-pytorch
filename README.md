# PredRNN++
This is a PyTorch implementation of [PredRNN](https://papers.nips.cc/paper/6689-predrnn-recurrent-neural-networks-for-predictive-learning-using-spatiotemporal-lstms), a deep recurrent predictive model for video data:

**PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs**, by Yunbo Wang, Mingsheng Long, Jianmin Wang, Zhifeng Gao, and Philip S. Yu.

## Setup
Required python libraries: pytorch + opencv + numpy.\

## Datasets
We conduct experiments on datasets [Moving Mnist](https://1drv.ms/f/s!AuK5cwCfU3__fGzXjcOlzTQw158) and [KTH Actions](http://www.nada.kth.se/cvap/actions/).\

## Training
Use the bash script to train the model:
```
cd script/
sh predrnn_mnist_train.sh
```
The learned model will be saved in the `--save_dir` folder.

## Test
The generated future frames will be saved in the `--gen_frm_dir` folder.

## Citation
Remember to cite our paper if you use the repository.
```
@inproceedings{wang2017predrnn,
  title={Predrnn: Recurrent neural networks for predictive learning using spatiotemporal lstms},
  author={Wang, Yunbo and Long, Mingsheng and Wang, Jianmin and Gao, Zhifeng and Philip, S Yu},
  booktitle={Advances in Neural Information Processing Systems},
  pages={879--888},
  year={2017}
}
```