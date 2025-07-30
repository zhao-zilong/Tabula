# Tabula 
The code is for paper [TabuLa: Harnessing Language Models for Tabular Data Synthesis](https://arxiv.org/abs/2310.12746). Tabula improves tabular data synthesis by leveraging language model 
structures without the burden of pre-trained model weights. It offers a faster training process by preprocessing tabular data to shorten token sequence, which 
sharply reducing training time while consistently delivering higher-quality synthetic data.
## Prerequisite

Tabula requires Python version >= 3.9, we have need the library versions to be:
```
datasets >= 2.5.2
numpy >= 1.24.2
pandas >= 1.4.4
scikit_learn >= 1.1.1
torch >= 1.10.2
tqdm >= 4.64.1
transformers >= 4.22.1
```

## Tabula quickstart  
Follow the python notebook `Tabula_on_insurance_dataset.ipynb` for a training example with Insurance dataset. The Insurance dataset is also provided within the code. We do not
hold the copyright of the dataset, the original dataset can also be download [here](https://www.kaggle.com/datasets/mirichoi0218/insurance). To download the pre-trained model on all datasets used in the paper, 
download [here](https://drive.google.com/file/d/1_YxelekxY5MXhgn93MYgsZEEfBYAy7h6/view?usp=sharing). Do not forget 
to create a folder `pretrained-model` and put the downloaded model inside.

## Problems about generation
### max_length issue
Since we published the code, we received several emails or open issues on the generation using Tabula folder. The problem is related to the **max_length** argument in `sample` function. The default length is 100, but the encoded data is longer than 100 tokens for a dataset with a large number of columns. In that case, users need to increase this number to allow Tabula to generate enough length of tokens for each row.
### column name issue
Please make sure your column name does not contain spaces. If you have a column named "A B", please rename it "A_B".

## Acknowledgement

Our code adapts the training structure of [GReaT](https://github.com/kathrinse/be_great/tree/main). Also thanks HuggingFace for their LLM model. 

## Citation

Please use following bibtex to cite this paper:
```
@inproceedings{zhao2025tabula,
  title={Tabula: Harnessing language models for tabular data synthesis},
  author={Zhao, Zilong and Birke, Robert and Chen, Lydia Y},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={247--259},
  year={2025},
  organization={Springer}
}
```
