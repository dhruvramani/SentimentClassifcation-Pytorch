import torch
from dataset import SST_Data
import torch.nn.functional as F
from embedding import WordEmbedding
from torch.autograd import Variable

class ConvNLP(torch.nn.Module):

    def __init__(self, vocab, no_classes):
        super(ConvNLP, self).__init__()

        self.vocab = vocab
        self.voc_size = len(vocab)
        self.no_classes = no_classes
        self.embedding_dim = 128
        self.conv_dim = 100

        self.embedding = WordEmbedding(self.vocab, self.embedding_dim)

        self.conv1 = torch.nn.Conv2d(1, self.conv_dim, [3, self.embedding_dim], (2, 0))
        self.conv2 = torch.nn.Conv2d(1, self.conv_dim, [4, self.embedding_dim], (3, 0))
        self.conv3 = torch.nn.Conv2d(1, self.conv_dim, [5, self.embedding_dim], (4, 0))

        self.dropout = torch.nn.Dropout(p=0.2)
        self.fc1 = torch.nn.Linear(self.conv_dim * 3, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, self.no_classes)

    def forward(self, X):
        embedding = self.embedding.embedAndPack(X, batch_first=True)
        embedding = torch.unsqueeze(embedding, 1)  # Channels is second dim, not last

        act1 = F.relu(self.conv1(embedding))
        act2 = F.relu(self.conv2(embedding))
        act3 = F.relu(self.conv3(embedding))

        acts = [act1, act2, act3]

        for i in range(len(acts)):
            acts[i] = torch.squeeze(acts[i], -1)
            acts[i] = F.max_pool1d(acts[i], acts[i].size(2))

        acts = torch.cat(acts, 2)
        acts = acts.view(acts.size(0), -1) # [batch_size, self.conv_dim * 3]

        act1 = self.dropout(self.fc1(acts))
        act2 = self.dropout(self.fc2(act1))
        act3 = self.fc3(act2)
        y_pred = F.softmax(act3)
        classes = torch.max(y_pred, 1)[1]

        return y_pred, classes

if __name__ == '__main__':
    dummy = SST_Data()
    data = dummy.get_data("train")
    data = next(data)[0]["text"]

    print(data)
    #sentence = torch.LongTensor(data)
    #print(sentence)
    #sentence = Variable(sentence, requires_grad=False)
    model = ConvNLP(dummy.get_vocab(), 5)
    print(model.forward(data))