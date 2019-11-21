from model import ErsatzTrainDataset, TransformerModel
import time
import torch
import torch.nn as nn
import argparse
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

class ErsatzTrainer():
    
    def __init__(self, train_path, batch_size, output_path, lr=5.0, cpu=False):
        
        self.with_cuda = torch.cuda.is_available() and not cpu
        self.device = torch.device("cuda" if self.with_cuda else "cpu")
        self.output_path = output_path       

        self.training_set = ErsatzTrainDataset(train_path, self.device)
        self.training_set.build_vocab()
        
        self.batch_size = batch_size
        self.training_set.batchify(batch_size)
        
        context_size = (self.training_set.window_size * 2) + 1
        self.model = TransformerModel(self.training_set.vocab, context_size).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        
        if self.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUSs for ET" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model) 

    def run_epoch(self, epoch, writer, batch_size):
        
        start = time.time()  
        total_tokens = 0
        total_loss = 0
        tokens = 0
        correct = 0
        total = 0
        for i, batch in enumerate(self.training_set.batches):
            
            data = batch.contexts.to(self.device)
            labels = batch.labels.to(self.device)
            output = self.model.forward(data)
            loss = self.criterion(output, labels)
            print(output.size())
            print(labels.size())
            pred = output.argmax(1)
            xor = pred ^ labels
            xor[xor==0] = -1
            xor[xor!=-1] = 0
            print(xor.size())
            correct += torch.sum(xor) 
            total += len(xor)    

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss
            total_tokens += batch.size * batch_size
            tokens += batch.size * batch_size
            if i % 50 == 1:
                time_mark = epoch*(len(self.training_set.batches)*batch_size) + i
                elapsed = time.time() - start
                writer.add_scalar('AverageLoss/train', loss/batch.size, time_mark)
                writer.add_scalar('Speed/train', tokens / elapsed)
                writer.add_scalar('Accuracy/train', correct/total)
                start = time.time()
                tokens = 0
        return total_loss / total_tokens


def train(self, epochs):
    for epoch in range(epochs):
        self.model.train()
        self.run_epoch()
        torch.save(self.model, self.output_path + 'e' + str(epoch))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path')
    parser.add_argument('--batch_size')
    parser.add_argument('--epochs')
    parser.add_argument('--output')
    parser.add_argument('--lr')
    parser.add_argument('--embed_size')
    parser.add_argument('--nhead')
    parser.add_argument('--nlayers')

    args = parser.parse_args()
    trainer = ErsatzTrainer(args.train_path, args.batch_size, args.output, args.lr, args.embed_size, args.nhead, args.nlayers) 

    writer = SummaryWriter()    

    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        trainer.train()
        trainer.run_epoch(epoch, writer, args.batch_size)
        torch.save(trainer, trainer.output_path + 'e' + str(epoch))
    best_model = trainer.train(args.epochs)
    torch.save(best_model, args.output)
