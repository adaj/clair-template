# ConSent

ConSent lets you train, evaluate and use deep learning models to classify chat dialogues, i.e., sequences of text that contains meta information such as *username*, *group*, *timestamp*. The use case in which ConSent was designed consists of classifying student chats, from different dialogues (group), according to a coding scheme. To train a model, one needs chat data coded in a particular coding scheme (we suggest at least 5k messages).

 > ConSent is compatible Keras and Tensorflow (we recommend version 2.5.0).

Below is the neural network framework we are using. We separate contextual information and sentence encoding features in two branches. Here, the sentence encoder means any pretrained language model capable of extracting vectors from text. This way, we benefit from the increasingly high potential of transfer learning to NLP tasks to produce more reliable estimates.

<img src="https://drive.google.com/uc?id=1aGcFBylS-KJjyrVMPG0JxBOrjTGExQLF" width="500">


## Installation

It is recommended to install it in a conda environment.
```
conda create --name consent python=3.9
conda activate consent
```

ConSent is not on PyPi. You can easily install it by cloning the repository to your working directory and install it locally from source.
```
cd <your directory>
git clone https://github.com/adaj/consent.git
cd consent
pip install --user -r requirements.txt 
pip install --user .
```

## How to use

### 1. Training a new model

Under construction

### 2. Testing your model

Under construction

### 3. Finding better parameters with the tuning experiment

Under construction

### 4. Evaluate with K-Fold cross-validation

Under construction

### 5. Check the outputs

Under construction


## Examples

Find examples in `examples/` and `paper/`.


## How to contribute

We are very happy to receive and merge your contributions into this repository!

To contribute via pull request, follow these steps:

1.  Check if there is an open issue describing the feature.
2.  Create an issue describing the feature you want to work on.
3.  Write your code, tests and documentation, and format them with  `black`.
4.  Create a pull request describing your changes.

Your pull request will be reviewed by a maintainer, who will get back to you about any necessary changes or questions. By sending a pull request, you agree with this repository's license agreement.

## Citation

BibTex:
```
@article{araujo2022consent,
  title={Automatic Coding of Student Chats with Contextual Information and Sentence Encoding},
  author={Araujo, Adelson and Papadopoulos, Pantelis and McKenney, Susan and de Jong, Ton},
  journal={Under Review},
  volume={},
  number={},
  pages={},
  year={2022},
  publisher={}
}
```

## License

Licensed under GNU General Public License v3.0.  Copyright 2022 Adelson D. de Araujo jr. [Copy of the license](https://github.com/adaj/consent/blob/master/LICENSE.md).

