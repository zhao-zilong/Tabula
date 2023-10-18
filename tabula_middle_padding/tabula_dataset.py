import random
import typing as tp

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
import torch


class TabulaDataset(Dataset):
    """ Tabula Dataset

    The TabulaDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """
    def set_tokenizer(self, tokenizer):
        """ Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer


    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        """ Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        idx_range = list(range(row.num_columns))
        # random.shuffle(shuffle_idx)
        data_row_text_list = []
        for i in idx_range:
            if i == 0:
                data_row_text_list.append("%s %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()))
            # elif i == row.num_columns -1:
            #     # data_row_text_list.append("%s" % (str(row.columns[i].to_pylist()[0]).strip()))
            else:
                data_row_text_list.append("%s" % (str(row.columns[i].to_pylist()[0]).strip()))
        # shuffled_text = ", ".join(
        #     ["%s is %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in shuffle_idx]
        # )
        # print("data row text list: ", data_row_text_list)
        return self.tokenizer(data_row_text_list)['input_ids']
        
    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)

@dataclass
class TabulaDataCollator(DataCollatorWithPadding):

    """ Tabula Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids

    """

    def set_token_list_length(self, token_list_length):
        """ Set the Tokenizer

        Args:
            token_list_length: a list, each element represents the longest token sequence size for the corresponding column
        """
        self.token_list_length = token_list_length
    


    def __call__(self, features):

        padding_token = 220
        number_sentences = len(features)
        encoding_text = []
        for i in range(len(self.token_list_length)):
            sublist = [item[i] for item in features] # sublist is a list of all the encoding for one data column
            # padded = zip(*itertools.zip_longest(*sublist, fillvalue=50256)) # padding 50256 to the shorter sublist
            if i == (len(self.token_list_length) - 1): # pad 50256 for the last column
                for e in sublist:
                    e.extend([50256] * (self.token_list_length[i] - len(e)))
            else:
                for e in sublist: # pad 220 for the middle column
                    e.extend([padding_token] * (self.token_list_length[i] - len(e)))
            # padded_list =[list(ele) for ele in list(sublist)]
            encoding_text.append(sublist)

        encoded_text = []
        for i in range(number_sentences):
            sentence = [item[i] for item in encoding_text]
            temp = [item for sublist in sentence for item in sublist] # flatten the list
            encoded_text.append(temp)
            
        batch = {'input_ids': torch.Tensor(encoded_text).long(), 'attention_mask': torch.ones(torch.Tensor(encoded_text).shape).long()}
        batch["labels"] = batch["input_ids"].clone()
     
        return batch
