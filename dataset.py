import torch
from torch.autograd import Variable
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe

class SST_Data():

    def __init__(self, batch_size=64):

        self.batch_size = batch_size
        self.TEXT = data.Field()
        self.LABEL = data.Field(sequential=False)

        self.train, self.val, self.test = datasets.SST.splits(self.TEXT, self.LABEL, 
            fine_grained=True, train_subtrees=True)

        self.url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        self.TEXT.build_vocab(self.train, vectors=Vectors('wiki.simple.vec', url=self.url))
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (self.train, self.val, self.test), batch_size=self.batch_size, device=0)

    def get_data(self, ttype):
        to_iter = {"train" : self.train_iter, "val" : self.val_iter, "test" : self.val_iter}[ttype]

        for batch in to_iter:
            X, Y = batch.text, batch.label
            print(X.size(), Y.size())
            yield X, Y

if __name__ == '__main__':
    foo = SST_Data()
    X, Y = foo.get_data("train")
    print(X.size())
