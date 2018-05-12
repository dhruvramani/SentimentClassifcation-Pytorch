import torch
from torch.autograd import Variable
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

'''
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

        print('len(train)', len(self.train))

        f = FastText()
        self.TEXT.build_vocab(self.train, vectors=f)
        self.TEXT.vocab.extend(f)
        self.LABEL.build_vocab(self.train)

        #self.TEXT.build_vocab(self.train, vectors=[GloVe(name='840B', dim='300'), CharNGram(), FastText()])
        #self.LABEL.build_vocab(self.train)

        print('len(TEXT.vocab)', len(self.TEXT.vocab))
        print('TEXT.vocab.vectors.size()', self.TEXT.vocab.vectors.size())

        self.train_iter, _, _ = datasets.SST.iters(batch_size=self.batch_size)

        #print('train[0:batch_size)', [vars(self.train[i]) for i in range(1, 10)])

        batch = next(iter(self.train_iter))
        print(batch.text)
        print(batch.label)

    def get_data(self, ttype):
        to_iter = {"train" : self.train_iter, "val" : self.val_iter, "test" : self.val_iter}[ttype]

        for batch in to_iter:
            X, Y = batch.text, batch.label
            print(X.size(), Y.size())
            yield X, Y

if __name__ == '__main__':
    foo = SST_Data()
    #X, Y = foo.get_data("train")
    #print(X.size())
