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

    def __init__(self, batch_size=64):
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

        #self.train_iter, _, _ = datasets.SST.iters(batch_size=700)

        #print('train[0:batch_size)', [vars(self.train[i]) for i in range(1, 10)])

        #batch = next(iter(self.train_iter))
        #print(batch.text)
        #print(batch.label)


    def get_data(self, ttype):
        to_iter = {"train" : self.train, "val" : self.val, "test" : self.test}[ttype]

        batches = len(to_iter) / self.batch_size
        for batch in range(int(batches) - 1):
            data = [vars(to_iter[i]) for i in range(batch * self.batch_size, (batch + 1) * self.batch_size)]
            yield data

    def get_vocab(self):
        return self.TEXT.vocab

if __name__ == '__main__':
    foo = SST_Data()
    data = foo.get_data("train")
    print(next(data))
