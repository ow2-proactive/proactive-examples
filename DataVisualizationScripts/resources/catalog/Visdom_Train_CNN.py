import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable
from visdom import Visdom

visdom_endpoint = variables.get("ENDPOINT_VISDOM") if variables.get("ENDPOINT_VISDOM") else results[0].__str__()
print("VISDOM_ENDPOINT: ", visdom_endpoint)

if visdom_endpoint is not None:
    visdom_endpoint = visdom_endpoint.replace("http://", "")

(VISDOM_HOST, VISDOM_PORT) = visdom_endpoint.split(":")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 3)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--visdom_host', type=str, default=VISDOM_HOST,
                    help='IP of the visdom server')
parser.add_argument('--visdom_port', type=int, default=VISDOM_PORT,
                    help='IP port of the visdom server')
args = parser.parse_args()

print("Connecting to %s:%s" % (args.visdom_host, args.visdom_port))
viz = Visdom(server="http://"+args.visdom_host, port=args.visdom_port)
assert viz.check_connection()

win_epoch_loss = viz.line(Y=np.array([np.nan]), X=np.array([np.nan]),
                          opts=dict(
                                  xlabel='Iteration',
                                  ylabel='Loss',
                                  title='Training loss (per iteration)',
                                  ),
                          )
win_global_loss = viz.line(Y=np.array([np.nan]), X=np.array([np.nan]),
                           opts=dict(
                                  xlabel='Epoch',
                                  ylabel='Loss',
                                  title='Training loss (per epoch)',
                                  ),
                           )
win_test_loss = viz.line(Y=np.array([np.nan]), X=np.array([np.nan]),
                           opts=dict(
                                  xlabel='Epoch',
                                  ylabel='Loss',
                                  title='Test loss (per epoch)',
                                  ),
                           )
win_test_acc = viz.line(Y=np.array([np.nan]), X=np.array([np.nan]),
                           opts=dict(
                                  xlabel='Epoch',
                                  ylabel='Accuracy',
                                  title='Test accuracy (per epoch)',
                                  ),
                           )
win_train_log = viz.text("Training log:\n")
win_test_log = viz.text("Testing log:\n")


use_cuda = not args.no_cuda and torch.cuda.is_available()
print("Use CUDA? " + str(use_cuda))

device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data', train=True, download=True,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))
                 ])),
  batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data', train=False, transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))
                 ])),
  batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

epoch_loss_train = []
average_loss_train = 0
iteration_train = 0
def train(epoch):
    global iteration_train
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        iteration_train = iteration_train + 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # store loss values
        epoch_loss_train.append(loss.data.item())
        average_loss_train = sum(epoch_loss_train) / len(epoch_loss_train)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.data.item()))
            viz.text('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.data.item()), win=win_train_log, append=True)
            # plot loss per iteration
            viz.line(Y=np.array([average_loss_train]), X=np.array([iteration_train]), win=win_epoch_loss, update='append')

    # plot loss per epoch
    viz.line(Y=np.array([average_loss_train]), X=np.array([epoch]), win=win_global_loss, update='append')


epoch_loss_test = []
average_loss_test = 0
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset), test_acc))
    viz.text('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset), test_acc), win=win_test_log, append=True)

    # plot loss per epoch
    viz.line(Y=np.array([test_loss]), X=np.array([epoch]), win=win_test_loss, update='append')
    viz.line(Y=np.array([test_acc]), X=np.array([epoch]), win=win_test_acc, update='append')


for epoch in range(1, args.epochs+1):
    train(epoch)
    test(epoch)