"""
Embedding models for a dataset of text.
"""


def embed_sm(dataset):
    """
    Embeds the text in a dataset using a simple embedding model.
    
    Parameters:
        dataset (dataset): A dataset of strings and labels. Each string is a sentence or phrase.
    """
    result = []

    return result


# class to tokenize the dataset
class TokenizedC4Dataset(Dataset):
    """
    A dataset that tokenizes the text in a dataset.
    """

    def __init__(self, dataset, encoding):
        self.dataset = dataset
        self.encoding = encoding

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        label = self.dataset[idx]["label"]

        text = self.encoding.encode(text)
        return text, label

    def preprocess_toktext_custom(self, text):
        """
        Preprocess the tokenized text for non-rnn models.
        """
        # if the list is less than 30, pad it with 0s
        if len(text) < 30:
            text = [0] * (30 - len(text)) + text

        # if the list is greater than 30, truncate it
        if len(text) > 30:
            text = text[:30]

        return text
