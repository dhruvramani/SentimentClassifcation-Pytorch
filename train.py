import torch 
from model import ConvNLP
from dataset import SST_Data
from torch.autograd import Variable

def train():
    learning_rate, epochs, curr_epoch, curr_step = 1e-4, 10, 0, 0
    dataobj = SST_Data()
    #data = dummy.get_data("train")
    model = ConvNLP(dataobj.get_vocab(), 5) # No. classes for SST
    
    loss_func = torch.nn.CrossEntropyLoss()
    optim =  torch.optim.AdamOptimizer(lr=learning_rate)

    if(torch.cuda.is_available()):
        model.cuda()

    while(curr_epoch < epochs):
        curr_step, avg_loss = 0, 0
        data_obj = SST_Data()
        for x_train, y_train in data_obj.get_data("train"):

            sentences = Variable(torch.LongTensor(x_train), requires_grad=False)
            labels = Variable(torch.LongTensor(y_train), requires_grad=False)

            if(torch.cuda.is_available()):
                sentences, labels = sentences.cuda(), labels.cuda()

            y_pred, classes = model(sentences)
            
            optim.zero_grad()
            loss = loss_func(y_pred, labels)
            loss.backward()
            optim.step()
            avg_loss += loss.data[0]
            curr_step += 1
        curr_epoch += 1
        print("Epoch {} - Avg Loss : {}".format(curr_epoch, avg_loss/curr_step))
