#  Semi-supervised Facial Affective Behavior Analysis (ABAW2021 submission)
*(Submission to the Affective Behavior Analysis in-the-wild ([ABAW](https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/)) 2021 competition)*

This repository presents a multi-task mean teacher model for semi-supervised Affective Behavior Analysis to learn from missing labels and exploring the learning of multiple correlated task simultaneously. 
Our method is modified from **[Two-Stream Aural-Visual Affect Analysis in the Wild](https://github.com/kuhnkeF/ABAW2020TNT)**. We have modified the preprocessing method of rendering mask to provide better semantics information. After that, we extended TSAV model to semi-supervised model using mean teacher, which allow it to be benefited from unlabeled data. 

For more detail, please check our paper.
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

Our paper have been submitted to arxiv and is in the queue to be announced.
You can find the pdf for our paper in this repository.

This repository is based on TSAV, thanks to their excellent work. 
Please cite their paper of TSAV if this respority helps you.

    @INPROCEEDINGS {,
    author = {F. Kuhnke and L. Rumberg and J. Ostermann},
    booktitle = {2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020) (FG)},
    title = {Two-Stream Aural-Visual Affect Analysis in the Wild},
    year = {2020},
    volume = {},
    issn = {},
    pages = {366-371},
    keywords = {expression recognition;action units;affective behavior analysis;human computer interaction;valence arousal;emotion recognition},
    doi = {10.1109/FG47880.2020.00056},
    url = {https://doi.ieeecomputersociety.org/10.1109/FG47880.2020.00056},
    publisher = {IEEE Computer Society},
    address = {Los Alamitos, CA, USA},
    month = {may}
    }




