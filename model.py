import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, depth):
        super(Attention, self).__init__()
        self.query_linear = nn.Linear(input_size, hidden_size)
        self.key_linear = nn.Linear(input_size, hidden_size)
        self.value_linear = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.depth_linear = nn.Linear(hidden_size, depth)
    def forward(self, query, keys, values):
        query = self.query_linear(query)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        scores = torch.matmul(query, keys.transpose(-2, -1))
        scores = self.softmax(scores)
        attended_values = torch.matmul(scores, values)
        output = self.depth_linear(attended_values)
        return output

class AT_GRU(nn.Module):
    def __init__(self, num_layers, rnn_hidden_size, encoder_input_size, encoder_hidden_size, depth=300):
        super(AT_GRU, self).__init__()
        self.depth = depth
        self.num_layers = num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.embedding_lat = nn.Embedding(180, 16)
        self.embedding_lon = nn.Embedding(360, 16)
        self.embedding_sst = nn.Embedding(64, 16)
        self.embedding_date = nn.Embedding(8192, 16)
        self.attention = Attention(encoder_input_size, encoder_hidden_size, depth)
        self.line = nn.Linear(16 * 4 + 4, 300)
        self.dropout = nn.Dropout(p=0.1 )
        self.bi_gru = nn.GRU(input_size=1, hidden_size=rnn_hidden_size,num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden_size * 2 * depth, depth)
        self.flatten = nn.Flatten(1, -1)

    def forward(self, x):
        x0 = torch.cat((self.embedding_lat(torch.trunc(x[:, 0]).long()), torch.frac(x[:, 0]).unsqueeze(1)), 1)
        x1 = torch.cat((self.embedding_lon(torch.trunc(x[:, 1]).long()), torch.frac(x[:, 1]).unsqueeze(1)), 1)
        x2 = self.embedding_date(x[:, 2].long())
        x3 = torch.cat((self.embedding_sst(torch.trunc(x[:, 3]).long()), torch.frac(x[:, 3]).unsqueeze(1)), 1)
        x = torch.cat((x0, x1, x2, x3, x[:, -1].unsqueeze(1)), 1)
        attention_output = self.attention(x.float(), x.float(), x.float())
        x = x.to(torch.float32)
        x = self.line(x)
        x = x + attention_output
        x = self.dropout(x)
        output, h = self.bi_gru(x.unsqueeze(-1))
        output = self.flatten(output)
        output = self.fc(output)
        return output