# Tabula 
The code is for paper [TabuLa: Harnessing Language Models for Tabular Data Synthesis](https://arxiv.org/abs/2310.12746). Tabula improves tabular data synthesis by leveraging language model 
structures without the burden of pre-trained model weights. It offers a faster training process by preprocessing tabular datato shorten token sequence, which 
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
hold the copyright of the dataset, the original dataset can also be download [here](https://www.kaggle.com/datasets/mirichoi0218/insurance). To download the pre-trained model on intrusion dataset as used in the paper. 
Download [here](https://drive.google.com/file/d/1BJ9shdCzOyMaXClB8oSIzyfjdjvjP8b-/view?usp=share_link). Do not forget 
to create a folder `pretrained-model` and put the downloaded model inside.

## Acknowledgement

Our code adapts the training structure of [GReaT](https://github.com/kathrinse/be_great/tree/main). Also thanks HuggingFace for their LLM model. 

## Citation

Please use following bibtex to cite this paper:
```
@misc{zhao2023tabula,
      title={TabuLa: Harnessing Language Models for Tabular Data Synthesis}, 
      author={Zilong Zhao and Robert Birke and Lydia Chen},
      year={2023},
      eprint={2310.12746},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```