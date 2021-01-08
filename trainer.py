from model import ErsatzTransformer
from dataset import ErsatzDataset
import time
import torch
import torch.nn as nn
import argparse
import torchvision
from determiner import PunctuationSpace, Split, MultilingualPunctuation
import os
import logging
import math
import json

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)

#global_determiner = Split()
global_determiner = PunctuationSpace()
#global_determiner = MultilingualPunctuation()


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


def load_model(checkpoint_path):
    model_dict = torch.load(checkpoint_path)
    model = ErsatzTransformer(model_dict['vocab'], model_dict['args'])
    model.load_state_dict(model_dict['weights'])
    return model

def save_model(model, output_path):
    model_dict = {
        'weights': model.state_dict(),
        'vocab': model.vocab,
        'args': model.args
    }
    torch.save(model_dict, output_path)


class ErsatzTrainer():
    
    def __init__(self, args):
        self.with_cuda = torch.cuda.is_available() and not args.cpu
        self.device = torch.device("cuda:0" if self.with_cuda else "cpu")
        self.output_path = args.output_path       
        self.batch_size = args.batch_size
        
        if os.path.exists(args.train_corpus_path):
            logging.info('Loading pre-processed training corpus')
            self.training_set = torch.load(args.train_corpus_path, map_location=torch.device('cpu'))
        else:
            logging.info('Building dataset...')
            self.training_set = ErsatzDataset(args.train_path, self.device, vocabulary_path=args.vocabulary_path, left_context_size=args.left_size, right_context_size=args.right_size)
            torch.save(self.training_set, args.train_corpus_path)

        if os.path.exists(args.valid_corpus_path):
            logging.info('Loading pre-processed validation corpus')
            self.validation_set = torch.load(args.valid_corpus_path, map_location=torch.device('cpu'))
        else:
            logging.info('Builing validation set...')
            self.validation_set = ErsatzDataset(args.valid_path, self.device, vocab=self.training_set.vocab, left_context_size=args.left_size, right_context_size=args.right_size)
            torch.save(self.validation_set, args.valid_corpus_path)
    
        left_context_size = self.training_set.left_context_size
        right_context_size = self.training_set.right_context_size

        logging.info(f'{self.device}')
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
        
        if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
            logging.info('Loading pre-existing model from checkpoint')
            self.model = torch.load(args.output_path, map_location=self.device)
        else:
            self.model = ErsatzTransformer(self.training_set.vocab, args).to(self.device)
            #self.model = ErsatzTransformer(self.training_set.vocab, left_context_size, right_context_size, embed_size=args.embed_size, nhead=args.nhead, num_layers=args.nlayers, dropout=args.dropout).to(self.device)
    
        weights = torch.tensor([args.eos_weight, 1]).to(self.device) 
        self.criterion = nn.NLLLoss(weight=weights)
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        total_params = sum([p.numel() for p in self.model.parameters()])
        logging.info(f'Training with: {total_params}')
        if self.with_cuda and torch.cuda.device_count() > 1:
            logging.info("Using %d GPUSs for ET" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model) 
            self.model = self.model.cuda()

    def validate(self, batch_size, determiner):
        retVal = {}
        retVal['num_obs_eos'] = 0
        retVal['num_pred_eos'] = 0
        retVal['correct_eos'] = 0
        retVal['correct_mos'] = 0
        retVal['total_loss'] = 0
        retVal['num_pred'] = 0
        retVal['inference_correct_eos'] = 0
        retVal['inference_incorrect_eos'] = 0
        retVal['inference_correct_mos'] = 0
        retVal['inference_incorrect_mos'] = 0
        self.model.eval()
        eos_ind = 0
        mos_ind = 1
        with torch.no_grad():
            for i, batch in self.validation_set.batchify(batch_size):
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
                for context_item, label_item, p in zip(batch.contexts, batch.labels, torch.argmax(output, dim=1)):
                    left_context = self.model.vocab.tensor_to_string(context_item[:self.model.left_context_size])
                    left_context = self.model.vocab.detokenize(left_context)

                    right_context = self.model.vocab.tensor_to_string(context_item[-self.model.right_context_size:])
                    right_context = self.model.vocab.detokenize(right_context)
                    if determiner(left_context, right_context):
                        if label_item == 0:
                            if p.item() == 0: 
                                retVal['inference_correct_eos'] += 1
                            else:
                                retVal['inference_incorrect_eos'] += 1
                        else:
                            if p.item() == 1:
                                retVal['inference_correct_mos'] += 1
                            else:
                                retVal['inference_incorrect_mos'] += 1

        if retVal['inference_correct_eos'] + retVal['inference_incorrect_mos'] != 0:
            retVal['inference_prec'] = retVal['inference_correct_eos']/(retVal['inference_correct_eos'] + retVal['inference_incorrect_mos'])
        else:
            retVal['inference_prec'] = 0
        if retVal['inference_correct_eos'] + retVal['inference_incorrect_eos'] != 0:
            retVal['inference_recall'] = retVal['inference_correct_eos']/(retVal['inference_correct_eos'] + retVal['inference_incorrect_eos'])
        else:
            retVal['inference_recall'] = 0
        if retVal['inference_prec'] != 0 and retVal['inference_recall'] != 0:
            retVal['inference_f1'] = 2*((retVal['inference_prec']*retVal['inference_recall'])/(retVal['inference_prec']+retVal['inference_recall']))
        else:
            retVal['inference_f1'] = 0

        retVal['inference_acc'] = (retVal['inference_correct_eos'] + retVal['inference_correct_mos'])/(retVal['inference_correct_eos'] + retVal['inference_correct_mos'] + retVal['inference_incorrect_eos'] + retVal['inference_incorrect_mos'])
        retVal['average_loss'] = retVal['total_loss']/retVal['num_pred']
        self.model.train()
        return retVal

    def run_epoch(self, epoch, writer, batch_size, log_interval, validation_interval, results, best_model, min_epochs = 10, validation_threshold=10):

        eos_ind = 0
        mos_ind = 1    
        for i, batch in self.training_set.batchify(batch_size):
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
            
                #writer.add_scalar('AverageLoss/train', status['average_loss'], status['time_passed'])
                #writer.add_scalar('Accuracy/train', status['acc'], status['time_passed'])
                #writer.add_scalar('Precision/train', status['prec'], status['time_passed'])
                #writer.add_scalar('Recall/train', status['recall'], status['time_passed'])
                #writer.add_scalar('F1/train', status['f1'], status['time_passed'])
                #writer.add_scalar('WPS', status['predictions_per_second'], status['time_passed'])
            
                results.reset(time.time())

            if i % validation_interval == 1:
                stats = self.validate(batch_size, global_determiner)
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
                #writer.add_scalar('AverageLoss/valid', stats['average_loss'], time_mark)
                #writer.add_scalar('Accuracy/valid', stats['acc'], time_mark)
                #writer.add_scalar('Precision/valid', stats['prec'], time_mark)
                #writer.add_scalar('Recall/valid', stats['recall'], time_mark)
                #writer.add_scalar('F1/valid', stats['f1'], time_mark)
                if best_model is not None:
                    if stats['inference_f1'] > best_model['inference_f1']:
                        save_model(self.model, os.path.join(self.output_path, 'checkpoint.best'))
                        best_model = stats
                        best_model['validation_num'] = status['validations']
                        logging.info(f'SAVING MODEL: { json.dumps(best_model)}')
                    else:
                        if epoch > min_epochs and status['validations'] - best_model['validation_num'] >= validation_threshold:
                            logging.info(f'EARLY STOPPING {json.dumps(best_model)}')
                            return 0, status, best_model
                else:
                    save_model(self.model, os.path.join(self.output_path, 'checkpoint.best'))
                    best_model = stats
                    logging.info(f'SAVING MODEL: { json.dumps(best_model) }')
                    best_model['validation_num'] = status['validations']
                    self.model.to(self.device)
        logging.info(f'SAVING MODEL: End of epoch {epoch}')
        save_model(self.model, os.path.join(self.output_path, f'checkpoint.e{epoch}')) 
        self.model.to(self.device) 
        return 1, status, best_model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path')
    parser.add_argument('valid_path')
    parser.add_argument('--vocabulary_path')
    parser.add_argument('--determiner_type', default='punc', choices=["punc", "split", "multilingual"])
    parser.add_argument('--left_size', type=int, default=15)
    parser.add_argument('--right_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--min-epochs', type=int, default=25)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--output_path', type=str, default='models')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--transformer_nlayers', type=int, default=2)
    parser.add_argument('--linear_nlayers', type=int, default=0)
    parser.add_argument('--activation_type', type=str, default="tanh", choices=["tanh"])
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--validation_interval', type=int, default=25000)
    parser.add_argument('--early_stopping', type=int, default=25)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--train_corpus_path', type=str, default='processed.train.corpus')
    parser.add_argument('--valid_corpus_path', type=str, default='processed.valid.corpus')
    parser.add_argument('--eos_weight', type=float, default=9.0)
    parser.add_argument('--seed', type=int, default=14)
    args = parser.parse_args()


    if args.determiner_type == "punc":
        global_determiner = PunctuationSpace()
    elif args.determiner_type == 'multilingual':
        global_determiner = MultilingualPunctuation()
    else:
        global_determiner = Split()


    torch.manual_seed(args.seed)
    logging.info('Starting trainer...')
    trainer = ErsatzTrainer(args)

    logging.info(trainer.model)
    logging.info(args)
    writer = None
    logging.info('Starting training...')
    minloss = math.inf
    status = {}
    status['type'] = 'TRAINING'
    best_model = None
    results = Results(time.time())
    for epoch in range(args.max_epochs):
        status['epoch'] = epoch
        trainer.model.train()
        res, status, best_model = trainer.run_epoch(epoch, writer, args.batch_size, args.log_interval, args.validation_interval, results, best_model, min_epochs=args.min_epochs, validation_threshold=args.early_stopping)
        if res == 0 and epoch > args.min_epochs:
            break
        trainer.scheduler.step()
