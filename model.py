import torch
import torch.nn.functional as F
from torch.Autograd import Variable

class ConvNLP(torch.nn.Module):

    def __init__(self, voc_size, no_classes):
        super(ConvNLP, self).__init__()

        self.voc_size = voc_size
        self.no_classes = no_classes
        self.embedding_dim = 128
        self.conv_dim = 100

        self.embedding = torch.nn.Embedding(self.voc_size, self.embedding_dim)

        self.conv1 = torch.nn.Conv2d(1, self.conv_dim, [3, embedding_dim], (2, 0))
        self.conv2 = torch.nn.Conv2d(1, self.conv_dim, [4, embedding_dim], (3, 0))
        self.conv3 = torch.nn.Conv2d(1, self.conv_dim, [5, embedding_dim], (4, 0))

        self.fc = torch.nn.Linear(self.conv_dim * 3, self.no_classes)

    def forward(self, X):
        embedding = self.embedding(X)
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

        y_pred = F.softmax(self.fc(acts))
        classes = torch.max(y_pred, 1)[1]

        return y_pred, classes

'''
class CLSTM(torch.nn.Module):

    def __init__(self, voc_size, no_classes):
        super(CLSTM, self).__init__()
        self.voc_size = voc_size
        self.no_classes = no_classes
'''
