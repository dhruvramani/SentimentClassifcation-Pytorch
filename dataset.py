import torch
from torch.autograd import Variable
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

'''
    Stanford Sentiment Treebank—an extension of MR but with train/dev/test splits provided and 
    fine-grained labels (very positive, positive, neutral, negative, very negative),
    re-labeled by Socher et al. (2013).

	For Docker, run :
		pip install torchtext
		pip install nltk
		apt-get update --fix-missing && apt-get locales
		locale-gen en_US.UTF-8
		export LANG="en_US.UTF-8"
		export LC_ALL="en_US.UTF-8"
	
'''

class SST_Data():

    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.TEXT = data.Field()
        self.LABEL = data.Field(sequential=False)

        self.train, self.val, self.test = datasets.SST.splits(self.TEXT, self.LABEL, 
            fine_grained=True, train_subtrees=True)

        f = FastText()
        self.TEXT.build_vocab(self.train, vectors=f)
        self.TEXT.vocab.extend(f)
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (self.train, self.val, self.test), batch_size=self.batch_size, device=1)


    '''
    def get_data(self, ttype):
        to_iter = {"train" : self.train, "val" : self.val, "test" : self.test}[ttype]

        batches = len(to_iter) / self.batch_size
        for batch in range(int(batches) - 1):
            data = [vars(to_iter[i]) for i in range(batch * self.batch_size, (batch + 1) * self.batch_size)]
            yield data
    '''

    def get_data(self, ttype):
        to_iter = {"train" : self.train_iter, "val" : self.val_iter, "test" : self.test_iter}[ttype]

        for batch in to_iter:
            X, Y = batch.text, batch.label
            print(X.size(), Y.size())
            yield X, Y

    def get_vocab(self):
        return self.TEXT.vocab

    def get_vocab_shape(self):
        return len(self.TEXT.vocab), self.TEXT.vocab.vectors.size()

if __name__ == '__main__':
    foo = SST_Data()
    data = foo.get_data("train")
    print(next(data)[0])
