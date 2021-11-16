from sklearn import metrics
import torch
import torch.optim as optim
import numpy as np
import time
import copy
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, f1_score, recall_score, average_precision_score
from utils import calc_metrics, evaluate_ffpp

class Trainer():
    def __init__(self, args, model,optimizer, scheduler, dataloader, criterion, logger, board_writer):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.logger = logger
        self.board_writer = board_writer
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.state_dict = {'key', 'value'}
    
    ### video-level
    def train_ffpp_video(self):
        start_time = time.time()
        for epoch in range(self.args.num_epochs):
            # train
            self.train_ffpp_video_epoch(epoch)
            # validation
            if epoch % self.args.val_every == 0:
                self.val_ffpp_video(epoch)
            
        end_time = time.time()
        self.logger.info('Training is finished. time: {:.0f}h {:.0f}m {:.0f}s'.format((end_time-start_time)//3600, ((end_time-start_time)%3600)//60, (end_time-start_time)%60))
        # 'saving model'
        print('Saving the final and the best model state dict...')
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir,self.args.model_name, self.args.exp_name,'model/{}_{}_epoch_{}.pth'.format(self.args.model_name, self.args.dataset, epoch)))
        torch.save(self.best_model_wts, os.path.join(self.args.save_dir, self.args.model_name, self.args.exp_name,'model/best_{}_{}_epoch_{}.pth'.format(self.args.model_name, self.args.dataset, self.best_epoch)))
        self.board_writer.close()
        # evaluation
        print('Test of final epoch model...')
        ACC, AUC, r_acc, f_acc = evaluate_ffpp(model=self.model, dataloader=self.dataloader['test'], device=self.args.device)
        self.logger.info('The final epoch model evaluation ACC:{:.4f}, AUC:{:.4f}, r_acc:{:.4f}, f_acc:{:.4f}'.format(ACC, AUC, r_acc, f_acc))
        self.model.load_state_dict(self.best_model_wts)
        print('Test of the best val model...')
        ACC, AUC, r_acc, f_acc = evaluate_ffpp(model=self.model, dataloader=self.dataloader['test'], device=self.args.device)
        self.logger.info('The best model evaluation ACC:{:.4f}, AUC:{:.4f}, r_acc:{:.4f}, f_acc:{:.4f}'.format(ACC, AUC, r_acc, f_acc))

    def train_ffpp_video_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_image_num = 0
        y_true, y_pred, y_score = [], [], []
        for batch_id, sample in enumerate(tqdm(self.dataloader['train'])):
            batch_size = sample['clip'].shape[0]
            image_batch = sample['clip'].to(self.args.device)
            label_batch = sample['label'].to(self.args.device)
            # model output + loss
            outputs = self.model(image_batch)
            loss = self.criterion['CE'](outputs, label_batch)
            # compute grads + opt step
            self.optimizer['CE'].zero_grad()
            loss.backward()
            self.optimizer['CE'].step()
            # record loss 
            train_loss = train_loss + loss.item() * batch_size  # loss of a batch
            train_image_num = train_image_num + batch_size  # calculate the training images
            # collect for metrics
            value, preds = torch.max(outputs, 1)  # value, dim
            y_score.extend(outputs.sigmoid().tolist())
            y_true.extend(label_batch.tolist())
            y_pred.extend(preds.tolist())

        # adjust learning rate
        self.logger.info('epoch:{} learning rate = {}'.format(epoch, self.optimizer['CE'].param_groups[0]['lr']))
        if self.scheduler is not None:
            self.scheduler.step()
        epoch_loss = train_loss / train_image_num
        # compute metrics
        epoch_acc, epoch_AUC, r_acc, f_acc = calc_metrics(y_true, y_pred, y_score)
        # write to tensorboard
        self.board_writer.add_scalars('train', {'train_loss': epoch_loss, 'train_acc': epoch_acc, 'real_acc': r_acc, 'fake_acc':f_acc}, epoch)
        self.logger.info('epoch:{}/{} train_loss:{:.4f} train_ACC:{:.4f} train_AUC:{:.4f} real_ACC:{:.4f} fake_ACC:{:.4f}'.format(epoch, self.args.num_epochs, epoch_loss, epoch_acc, epoch_AUC, r_acc, f_acc))

    def val_ffpp_video(self, epoch):
        self.model.eval()
        val_image_num = 0
        val_loss = 0.0
        y_true, y_pred, y_score = [], [], []

        with torch.no_grad():
            for batch_id, sample in enumerate(tqdm(self.dataloader['val'])):
                batch_size = sample['clip'].shape[0]
                image_batch = sample['clip'].to(self.args.device)
                label_batch = sample['label'].to(self.args.device)
                # model output + loss
                outputs = self.model(image_batch)
                loss = self.criterion['CE'](outputs, label_batch)
                # record loss
                val_image_num = val_image_num + batch_size
                val_loss = val_loss + loss.item() * batch_size
                # collect for metrics
                value, preds = torch.max(outputs, 1)
                y_score.extend(outputs.sigmoid().tolist())
                y_true.extend(label_batch.tolist())
                y_pred.extend(preds.tolist())      

            val_loss = val_loss / val_image_num
            # compute metrics
            val_acc, val_AUC, r_acc, f_acc = calc_metrics(y_true, y_pred, y_score)

        if val_acc >= self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.board_writer.add_scalars('val', {'val_loss': val_loss, 'val_acc': val_acc}, epoch)
        self.logger.info('epoch:{} val_loss:{:.4f} val_ACC:{:.4f} val_AUC:{:.4f} real_ACC:{:.4f} fake_ACC:{:.4f}'.format(epoch, val_loss, val_acc, val_AUC, r_acc, f_acc))
        self.logger.info('current best model is epoch {} val_acc: {:.4f}'.format(self.best_epoch, self.best_acc))
        self.logger.info('--------------------------------------------------')