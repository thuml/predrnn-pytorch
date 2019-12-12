# PredRNN
This is a PyTorch implementation of [PredRNN](https://papers.nips.cc/paper/6689-predrnn-recurrent-neural-networks-for-predictive-learning-using-spatiotemporal-lstms), a deep recurrent predictive model for video data. Remember to cite our paper if you use the repository.
```
@inproceedings{wang2017predrnn,
  title={Predrnn: Recurrent neural networks for predictive learning using spatiotemporal lstms},
  author={Wang, Yunbo and Long, Mingsheng and Wang, Jianmin and Gao, Zhifeng and Philip, S Yu},
  booktitle={Advances in Neural Information Processing Systems},
  pages={879--888},
  year={2017}
}
```

## Steps
1. Install Python 3.7, PyTorch 1.3, and Opencv 3.4.  

2. Download data. This repo contains code for two datasets: the [Moving Mnist dataset](https://1drv.ms/f/s!AuK5cwCfU3__fGzXjcOlzTQw158) and the [human action KTH dataset](http://www.nada.kth.se/cvap/actions/).  

3. Train the model. You can use the bash script to train the model:
```
cd script/
sh predrnn_mnist_train.sh
```
The learned model will be saved in the `--save_dir` folder.  
The generated future frames will be saved in the `--gen_frm_dir` folder.

## Related Code Repo
**PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning.**  
Yunbo Wang, Zhifeng Gao, Mingsheng Long, Jianmin Wang, and Philip S. Yu.  
ICML 2018 [[paper](http://proceedings.mlr.press/v80/wang18b.html)] [[code](https://github.com/Yunbo426/predrnn-pp)]
