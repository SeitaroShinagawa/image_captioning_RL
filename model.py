
import torch
import torch.nn as nn
import torch.nn.parallel
from torch import optim
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.01)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun



class RNN_DECODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, n_steps=30, bidirectional=True, w_init='uniform'):
        super(RNN_DECODER, self).__init__()
        self.n_steps = n_steps # size of max word length
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = "GRU"
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights(w_init)

    def define_module(self):
        self.img_emb = nn.Linear(4096, self.nlayers * self.num_directions * self.nhidden) #(num_layers * num_directions * hidden_size)
        self.encoder = nn.Embedding(self.ntoken, self.ninput) 
        self.out = nn.Linear(self.nhidden * self.num_directions, self.ntoken)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self, w_init):
        assert w_init in ['uniform', 'gaussian']

        if w_init == 'uniform': 
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
        else:
            nn.init.normal_(self.encoder.weight, 0.0, 0.01)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        hidden = hidden.contiguous()
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len

        words_emb = output.transpose(1, 2)
        B, H, L = words_emb.shape

        y = self.out(output.reshape(B*L, H)) # (B*L, ntoken)
        y = F.log_softmax(y, dim=1)
        y = y.reshape(B, L, self.ntoken).transpose(1, 2) # (B, ntoken, L)

        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)

        return y, words_emb, sent_emb

    def embed_img(self, img_feat):
        a = self.drop(self.img_emb(img_feat)) # (batch, 4096) -> (batch, num_layers * num_directions * hidden_size)
        a = a.reshape(-1, self.nlayers * self.num_directions, self.nhidden)   # (batch, 4096) -> (batch, num_layers * num_directions, hidden_size)
        a = a.transpose(0, 1)
        return a

    def make_optimizer(self, lr=2e-4):
        return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            betas=(0.9, 0.999))


