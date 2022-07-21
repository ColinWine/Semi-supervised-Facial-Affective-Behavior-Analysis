#  Semi-supervised Facial Affective Behavior Analysis (ABAW2021 submission)
*(Submission to the Affective Behavior Analysis in-the-wild ([ABAW](https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/)) 2021 competition)*

This repository presents a multi-task mean teacher model for semi-supervised Affective Behavior Analysis to learn from missing labels and exploring the learning of multiple correlated task simultaneously. 

For more detail, please check our paper: [Arxiv](https://arxiv.org/abs/2107.04225).
## Required packages


torch                    1.6.0, 
torchaudio               0.6, 
tqdm, 
Numpy, 
OpenCV 4.2.0, 
lmdb


## Testing

To predict on competition test set, download our model and alignment files:  

[Alignment_face,Model Weight,Alignment data](https://pan.baidu.com/s/1dUmc8gVv9mlaWvAwQsH5gg) can be download here with extract code: wk36


Clone the repository,then download above data and config their path in opts.py before running 

    python test_val_aff2.py


## Citation

Our paper have been submitted to [Arxiv](https://arxiv.org/abs/2107.04225).



This repository is based on [TSAV](https://github.com/alexmehta/ABAW-TNT-Modified), thanks to their excellent work. Please cite their paper of TSAV if this respority helps you.




