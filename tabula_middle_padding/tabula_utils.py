import typing as tp

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer


def _array_to_dataframe(data: tp.Union[pd.DataFrame, np.ndarray], columns=None) -> pd.DataFrame:
    """ Converts a Numpy Array to a Pandas DataFrame

    Args:
        data: Pandas DataFrame or Numpy NDArray
        columns: If data is a Numpy Array, columns needs to be a list of all column names

    Returns:
        Pandas DataFrame with the given data
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), "Input needs to be a Pandas DataFrame or a Numpy NDArray"
    assert columns, "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    assert len(columns) == len(data[0]), \
        "%d column names are given, but array has %d columns!" % (len(columns), len(data[0]))

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    """ Returns the distribution of a given column. If continuous, returns a list of all values.
        If categorical, returns a dictionary in form {"A": 0.6, "B": 0.4}

    Args:
        df: pandas DataFrame
        col: name of the column

    Returns:
        Distribution of the column
    """
    if df[col].dtype == "float":
        col_dist = df[col].to_list()
    else:
        col_dist = df[col].value_counts(1).to_dict()
    return col_dist


def _convert_tokens_to_dataframe(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer, token_list_length: list, column_list: list, df_gen):
    
    result_list = []
    for t in tokens:
        token_list_cursor = 0
        td = dict.fromkeys(column_list)
        for idx, token_list_span in enumerate(token_list_length):
            decoded_text = tokenizer.decode(t[token_list_cursor:token_list_cursor+token_list_span])
            decoded_text = ("").join(decoded_text)
            # print("after combination: ", decoded_text)
            token_list_cursor = token_list_cursor + token_list_span
            # Clean text
            decoded_text = decoded_text.replace("<|endoftext|>", "")
            decoded_text = decoded_text.replace("\n", " ")
            decoded_text = decoded_text.replace("\r", "")
            # print("after cleaning: ", decoded_text)
            if idx == 0:
                values = decoded_text.strip().split(" ")
                try:
                    if len(values) > 1:
                        # print("key value: ", column_list[idx], values[1])
                        td[column_list[idx]] = [values[1]]
                except IndexError:
                    print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                    pass
            else:
                try:
                    # print("key value: ", column_list[idx], decoded_text)
                    td[column_list[idx]] = [decoded_text]
                except IndexError:
                    print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                    pass                
        # print("data frame: ", pd.DataFrame(td))
        result_list.append(pd.DataFrame(td))

    generated_df = pd.concat(result_list, ignore_index=True, axis=0)
    df_gen = pd.concat([df_gen, generated_df], ignore_index=True, axis=0)

    return df_gen




def _convert_tokens_to_text(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer) -> tp.List[str]:
    """ Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data



def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
    """ Converts the sentences back to tabular data

    Args:
        text: List of the tabular data in text form
        df_gen: Pandas DataFrame where the tabular data is appended

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    columns = df_gen.columns.to_list()
    result_list = []
    # Convert text to tabular data
    for t in text:
        features = t.split(",")
        td = dict.fromkeys(columns)
        
        # Transform all features back to tabular data
        for f in features:
            values = f.strip().split(" ")
            if values[0] in columns and not td[values[0]]:
                try:
                    td[values[0]] = [values[1]]
                except IndexError:
                    #print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                    pass
        result_list.append(pd.DataFrame(td))   
        # df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
    generated_df = pd.concat(result_list, ignore_index=True, axis=0)
    df_gen = pd.concat([df_gen, generated_df], ignore_index=True, axis=0)
    return df_gen