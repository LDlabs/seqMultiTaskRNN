# seqMultiTaskRNN
This repository contains code to accompany the [paper](https://proceedings.neurips.cc/paper/2020/hash/a576eafbce762079f7d1f77fca1c5cc2-Abstract.html)

Duncker, L.\*, Driscoll, L.\*, Shenoy, K. V., Sahani, M.\*\*, & Sussillo, D.\*\* (2020). Organizing recurrent network dynamics by task-computation to enable continual learning. Advances in Neural Information Processing Systems, 33.

```
@inproceedings{duncker+driscoll:2020:neurips,
  title={Organizing recurrent network dynamics by task-computation to enable continual learning},
  author={Duncker, Lea and Driscoll, Laura N and Shenoy, Krishna V and Sahani, Maneesh and Sussillo, David},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
the repository is based on work in

Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T., & Wang, X. J. (2019). Task representations in neural networks trained to perform many cognitive tasks. Nature neuroscience, 22(2), 297-306.

and code which can be found [here](https://github.com/gyyang/multitask) (though versions might have diverged)

## Installation
The code runs on Python 2.7 and an older tensorflow version. After cloning the repository, you can create a virtual environment and install the requirements using

```
virtualenv -p /usr/bin/python2.7 seqRNN
source seqRNN/bin/activate
pip install -r requirements.txt
```

## Examples
An example script for sequentially training an RNN on the task-set from the paper using our continual learning approach is provided in the script `example_sequential_training.py`


The folder `data/trained_models/` contains an example trained network. `analyses/demos.ipynb` contains some examples to reproduce analyses from the paper.

Some of the analyses rely on running [FixedPointFinder](https://github.com/mattgolub/fixed-point-finder) on the trained RNN.
You need to [install FixedPointFinder](https://github.com/mattgolub/fixed-point-finder) and then specify the relevant Python path in `analyses/run_fixed_point_finder.py` by editing the line

```
PATH_TO_FIXED_POINT_FINDER = '/path/to/your/directory/fixed-point-finder/'
```
to match the path corresponding to your local directory.
