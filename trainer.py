import model
import dataset
#from model import TransformerModel
#from dataset import ErsatzTrainDataset
import time
import torch
import torch.nn as nn
import argparse
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
import logging
import math
import json

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)

class Results():
    def __init__(self, time):
        self.total_loss = 0
        self.num_obs_eos = 0
        self.num_pred_eos = 0
        self.correct_eos = 0
        self.correct_mos = 0
        self.num_pred = 0
        self.update_num = 0
        self.last_update = time
        self.start = time
        self.validations = 0
        self.type = 'TRAINING'   
 
    def calculate(self, loss, predictions, labels, eos_ind):
        self.total_loss += loss
        self.num_pred += len(predictions)
        self.num_obs_eos += (labels==eos_ind).sum().item()
        self.num_pred_eos += (predictions==eos_ind).sum().item()

        predictions[predictions!=eos_ind] = -1
        predictions = predictions ^ labels
        self.correct_eos += (predictions==0).sum().item()
   
        self.correct_mos = self.num_pred - (self.num_obs_eos + self.num_pred_eos - self.correct_eos)            

    def get_results(self, lr):
        retVal = {}
        retVal['type'] = self.type
        retVal['update_num'] = self.update_num
        if self.num_pred_eos != 0:
            retVal['prec'] = self.correct_eos/self.num_pred_eos
        else:
            retVal['prec'] = 0
        if self.num_obs_eos != 0:
            retVal['recall'] = self.correct_eos/self.num_obs_eos
        else:
            retVal['recall'] = 0
        retVal['acc'] = (self.correct_eos + self.correct_mos)/self.num_pred
        if retVal['prec'] != 0 and retVal['recall'] != 0:
            retVal['f1'] = 2*((retVal['prec']*retVal['recall'])/(retVal['prec']+retVal['recall']))
        else:
            retVal['f1'] = 0
        retVal['lr'] = lr
        retVal['total_loss'] = self.total_loss
        retVal['average_loss'] = self.total_loss/self.num_pred
        retVal['time_since_last_update'] = time.time()-self.last_update
        retVal['predictions_per_second'] = self.num_pred/retVal['time_since_last_update']
        retVal['time_passed'] = time.time()-self.start
        retVal['correct_eos'] = self.correct_eos
        retVal['correct_mos'] = self.correct_mos
        retVal['num_pred_eos'] = self.num_pred_eos
        retVal['num_obs_eos'] = self.num_obs_eos
        retVal['validations'] = self.validations
        retVal['num_pred'] = self.num_pred
        return retVal

    def reset(self, time):
        self.total_loss = 0
        self.num_obs_eos = 0
        self.num_pred_eos = 0
        self.correct_eos = 0
        self.correct_mos = 0
        self.num_pred = 0
        self.update_num += 1
        self.last_update = time

    def validated(self):
        self.validations += 1

class ErsatzTrainer():
    
    def __init__(self, train_path, valid_path, batch_size, output_path, vocabulary_path, lr=5.0, embed_size=512, nhead=8, nlayers=6, cpu=False, train_corpus_path='processed.train.corpus', valid_corpus_path='processed.valid.corpus'):
        
        self.with_cuda = torch.cuda.is_available() and not cpu
        self.device = torch.device("cuda:0" if self.with_cuda else "cpu")
        self.output_path = output_path       
        self.batch_size = batch_size
        
        if os.path.exists(train_corpus_path):
            logging.info('Loading pre-processed training corpus')
            self.training_set = torch.load(train_corpus_path, map_location=torch.device('cpu'))
        else:
            logging.info('Building dataset...')
            self.training_set = dataset.ErsatzTrainDataset(train_path, self.device, batch_size, vocabulary_path)
            torch.save(self.training_set, train_corpus_path)

        if os.path.exists(valid_corpus_path):
            logging.info('Loading pre-processed validation corpus')
            self.validation_set = torch.load(valid_corpus_path, map_location=torch.device('cpu'))
        else:
            logging.info('Builing validation set...')
            self.validation_set = dataset.ErsatzValidDataset(valid_path, self.device, batch_size, self.training_set.vocab)
            torch.save(self.validation_set, valid_corpus_path)
    
        left_context_size = self.training_set.left_context_size
        right_context_size = self.training_set.right_context_size

        logging.info(f'{self.device}')
        if os.path.exists(output_path):
            logging.info('Loading pre-existing model from checkpoint')
            self.model = torch.load(output_path, map_location=self.device)
        else:
            self.model = model.TransformerModel(self.training_set.vocab, left_context_size, right_context_size, embed_size=embed_size, nhead=nhead, num_layers=nlayers).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
       
        print(self.model)
        total_params = sum([p.numel() for p in self.model.parameters()])
        print(f'Training with: {total_params}')
        if self.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUSs for ET" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model) 
            self.model = self.model.cuda()

    def validate(self, batch_size):
        retVal = {}
        retVal['num_obs_eos'] = 0
        retVal['num_pred_eos'] = 0
        retVal['correct_eos'] = 0
        retVal['correct_mos'] = 0
        retVal['total_loss'] = 0
        retVal['num_pred'] = 0
        self.model.eval()
        eos_ind = self.training_set.vocab.embed_word('<eos>')
        with torch.no_grad():
            for i, batch in enumerate(self.validation_set.batches):
                data = batch.contexts.to(self.device)
                labels = batch.labels.to(self.device)
                output = self.model.forward(data)
                loss = self.criterion(output, labels)
    
                pred = output.argmax(1)
        
                retVal['num_pred'] += len(pred)
                retVal['num_obs_eos'] += (labels==eos_ind).sum().item()
                retVal['num_pred_eos'] += (pred==eos_ind).sum().item()

                pred[pred!=eos_ind] = -1
                pred = pred ^ labels
                retVal['correct_eos'] += (pred==0).sum().item()

                retVal['correct_mos'] = retVal['num_pred'] - (retVal['num_obs_eos'] + retVal['num_pred_eos'] - retVal['correct_eos'])
        
                retVal['total_loss'] += loss.item()
        retVal['average_loss'] = retVal['total_loss']/retVal['num_pred']
        self.model.train()
        return retVal

    def run_epoch(self, epoch, writer, batch_size, log_interval, validation_interval, results, best_model, min_epochs = 10, validation_threshold=50):

        eos_ind = self.training_set.vocab.embed_word('<eos>')
        
        for file_num in range(self.training_set.file_count):
            batches = torch.load(f'{self.training_set.train_path}.{file_num}.bin', map_location=torch.device('cpu'))
            for i, batch in enumerate(batches): 
                data = batch.contexts.to(self.device)
                labels = batch.labels.to(self.device)
                output = self.model.forward(data)
                loss = self.criterion(output, labels)
           
                pred = output.argmax(1)
        
                results.calculate(loss.item(), pred, labels, eos_ind)
            
                self.optimizer.zero_grad() 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
            
                if i % log_interval == 1:
                    status = results.get_results(self.scheduler.get_lr()[0]) 
                    logging.info(json.dumps(status))
                
                    writer.add_scalar('AverageLoss/train', status['average_loss'], status['time_passed'])
                    writer.add_scalar('Accuracy/train', status['acc'], status['time_passed'])
                    writer.add_scalar('Precision/train', status['prec'], status['time_passed'])
                    writer.add_scalar('Recall/train', status['recall'], status['time_passed'])
                    writer.add_scalar('F1/train', status['f1'], status['time_passed'])
                    writer.add_scalar('WPS', status['predictions_per_second'], status['time_passed'])
                
                    results.reset(time.time())

                if i % validation_interval == 1:
                    stats = self.validate(batch_size)
                    stats['type'] = 'VALIDATION'
                    results.validated()
                    time_mark = time.time()-results.start
                    stats['average_loss'] = stats['total_loss']/stats['num_pred']
                    stats['acc'] = (stats['correct_eos'] + stats['correct_mos'])/stats['num_pred']
                    if stats['num_pred_eos'] != 0:
                        stats['prec'] = stats['correct_eos']/stats['num_pred_eos']
                    else:
                        stats['prec'] = 0
                    if stats['num_obs_eos'] != 0:
                        stats['recall'] = stats['correct_eos']/stats['num_obs_eos']
                    else:
                        stats['recall'] = 0
                    if stats['prec'] != 0 and stats['recall'] != 0:
                        stats['f1'] = 2*(stats['prec']*stats['recall'])/(stats['prec']+stats['recall'])
                    else:
                        stats['f1'] = 0
                    logging.info(json.dumps(stats))
                    writer.add_scalar('AverageLoss/valid', stats['average_loss'], time_mark)
                    writer.add_scalar('Accuracy/valid', stats['acc'], time_mark)
                    writer.add_scalar('Precision/valid', stats['prec'], time_mark)
                    writer.add_scalar('Recall/valid', stats['recall'], time_mark)
                    writer.add_scalar('F1/valid', stats['f1'], time_mark)
                    if best_model is not None:
                        if stats['average_loss'] < best_model['average_loss']:
                            torch.save(self.model.cpu(), self.output_path + '.best')
                            best_model = stats
                            self.model.to(self.device)
                            best_model['validation_num'] = status['validations']
                            logging.info(f'SAVING MODEL: { json.dumps(best_model)}')
                        else:
                            if epoch > min_epochs and status['validations'] - best_model['validation_num'] == validation_threshold:
                                logging.info(f'EARLING STOPPING {json.dumps(best_model)}')
                                return 0, status, best_model
                    else:
                        torch.save(self.model.cpu(), self.output_path + '.best')
                        best_model = stats
                        logging.info(f'SAVING MODEL: { json.dumps(best_model) }')
                        best_model['validation_num'] = status['validations']
                        self.model.to(self.device)
        return 1, status, best_model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path')
    parser.add_argument('valid_path')
    parser.add_argument('vocabulary_path')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--min-epochs', type=int, default=10)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--output')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--embed_size', type=int)
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--nlayers', type=int)
    parser.add_argument('--log_interval', type=int)
    parser.add_argument('--validation_interval', type=int)

    args = parser.parse_args()
    
    logging.info('Starting trainer...')
    trainer = ErsatzTrainer(args.train_path, args.valid_path, args.batch_size, args.output, args.vocabulary_path, args.lr, args.embed_size, args.nhead, args.nlayers) 

    writer = SummaryWriter()    
    logging.info('Starting training...')
    minloss = math.inf
    status = {}
    status['type'] = 'TRAINING'
    best_model = None
    results = Results(time.time())
    for epoch in range(args.max_epochs):
        status['epoch'] = epoch
        trainer.model.train()
        res, status, best_model = trainer.run_epoch(epoch, writer, args.batch_size, args.log_interval, args.validation_interval, results, best_model, min_epochs=args.min_epochs)
        if res == 0 and epoch > args.min_epochs:
            break
        trainer.scheduler.step()
