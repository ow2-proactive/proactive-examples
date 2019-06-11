print("BEGIN GRU")
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE=2
HIDDEN_DIM=50
EMBEDDING_DIM=50
DROPOUT=0.5

if 'variables' in locals():  
  if variables.get("HIDDEN_DIM") is not None:
    HIDDEN_DIM = variables.get("HIDDEN_DIM")
  else:
    print("HIDDEN_DIM not defined by the user. Using the default value:"+HIDDEN_DIM)
  if variables.get("EMBEDDING_DIM") is not None:
    EMBEDDING_DIM = variables.get("EMBEDDING_DIM")
  else:
    print("EMBEDDING_DIM not defined by the user. Using the default value:"+EMBEDDING_DIM)
  if variables.get("DROPOUT") is not None:
    DROPOUT = variables.get("DROPOUT")
  else:
    print("DROPOUT not defined by the user. Using the default value:"+DROPOUT)


MODEL_TYPE = 'GRU'
MODEL_CLASS = """
class GRU(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = len(train)
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.GRU = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        x = self.embeddings(sentence)
        gru_out, self.hidden = self.GRU(x, self.hidden)
        y = self.hidden2label(gru_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs"""
    
MODEL_DEF = """
MODEL = GRU(embedding_dim="""+str(EMBEDDING_DIM)+""", hidden_dim="""+str(HIDDEN_DIM)+""", vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,use_gpu=USE_GPU, batch_size=len(train))
"""
print(MODEL_DEF)

# Forward model
try:
    variables.put("MODEL_CLASS", MODEL_CLASS)
    variables.put("MODEL_DEF", MODEL_DEF)
    variables.put("HIDDEN_DIM", HIDDEN_DIM)
    variables.put("EMBEDDING_DIM", EMBEDDING_DIM)
    variables.put("DROPOUT", DROPOUT)
except NameError as err:
    print("{0}".format(err))
    print("Warning: this script is running outside from ProActive.")
    pass

print("END GRU")