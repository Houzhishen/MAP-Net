import torch
import torch.nn as nn
import torch.optim as optim
import shutil
#import seaborn as sns
import numpy as np
import os
import scipy as sp
import scipy.stats

from models.prototype import get_prototypes, prototypical_loss, get_proto_accuracy



class ModelTrainer(object):
    def __init__(self,
                args,
                enc_module,
                lp_module,
                data_loader,
                writer):

        self.args = args
        self.enc_module = enc_module.cuda()
        self.lp_module = lp_module.cuda()

        # get data loader
        self.data_loader = data_loader

        # set optimizer
        self.module_params = list(self.enc_module.parameters()) + list(self.lp_module.parameters()) 

        if self.args.optim == 'adam':
            self.optimizer = optim.Adam(params=self.module_params,
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)
        elif self.args.optim == 'sgd':
            self.optimizer = optim.SGD(params=self.module_params,
                                        lr=args.lr,
                                        momentum=0.9,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)


        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

        self.writer = writer


    def train(self):
        val_acc = self.val_acc

        batch_size = self.args.train_batch
        num_supports = self.args.nKnovel * self.args.nExemplars
        num_queries = self.args.train_nTestNovel
        num_samples = num_supports + num_queries

        # for each iteration
        for epoch in range(self.global_step + 1, self.args.max_epoch + 1):
    
            # set as train mode
            self.enc_module.train()
            self.lp_module.train()
            
            # set current step
            self.global_step = epoch
            train_len = len(self.data_loader['train'])
            for idx, (support_data, support_word, support_label, query_data, query_word_real, query_label, _) in enumerate(self.data_loader['train']):
                # print(i)
                support_data = support_data.cuda()
                support_word = support_word.cuda()
                support_label = support_label.cuda()
                query_data = query_data.cuda()
                query_word_real = query_word_real.cuda()
                query_label = query_label.cuda()

                query_word = torch.zeros(query_data.size(0), query_data.size(1), support_word.size(2)).cuda()

                # set as single data
                full_data = torch.cat([support_data, query_data], 1)

                # (1) encode data
                full_data_embeddings_view = self.enc_module(full_data.view(-1, *full_data.shape[2:]))
                full_data_embeddings = full_data_embeddings_view.view(*full_data.shape[:2], -1)

                support_data_embeddings = full_data_embeddings[:, :num_supports]
                query_data_embeddings = full_data_embeddings[:, num_supports:]

                # (2) propagation
                support_data_embeddings_tiled = support_data_embeddings.unsqueeze(1).repeat(1, num_queries, 1, 1)
                support_data_embeddings_tiled = support_data_embeddings_tiled.view(batch_size*num_queries, num_supports, -1)
                query_data_embeddings_tiled = query_data_embeddings.contiguous().view(batch_size*num_queries, -1).unsqueeze(1)

                support_word_tiled = support_word.unsqueeze(1).repeat(1, num_queries, 1, 1)
                support_word_tiled = support_word_tiled.view(batch_size*num_queries, num_supports, -1)
                query_word_tiled = torch.zeros(batch_size*num_queries, 1, query_word.size(2)).cuda()

                query_word_real_tiled = query_word_real.contiguous().view(batch_size*num_queries, -1).unsqueeze(1)

                support_label_tiled = support_label.unsqueeze(1).repeat(1, num_queries, 1)
                support_label_tiled = support_label_tiled.view(batch_size*num_queries, num_supports)
                query_label_tiled = query_label.contiguous().view(batch_size*num_queries, -1)

                full_data_embeddings_tiled = torch.cat([support_data_embeddings_tiled, query_data_embeddings_tiled], dim=1)
                full_word_tiled = torch.cat([support_word_tiled, query_word_tiled], dim=1)
                full_word_real_tiled = torch.cat([support_word_tiled, query_word_real_tiled], dim=1)

                embeddings, addition_loss = self.lp_module(full_data_embeddings_tiled, full_word_tiled, full_word_real_tiled)

                support_embeddings = embeddings[:, :num_supports, :]
                query_embeddings = embeddings[:, -1, :].unsqueeze(1)

                prototype = get_prototypes(support_embeddings, support_label_tiled, self.args.nKnovel)
                # print(prototype.size())
                # print(query_embeddings.size())

                loss_cls = prototypical_loss(prototype, query_embeddings, query_label_tiled)
                acc = get_proto_accuracy(prototype, query_embeddings, query_label_tiled)
                loss = loss_cls + addition_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # adjust learning rate
                self.adjust_learning_rate(optimizers=[self.optimizer],
                                        lr=self.args.lr,
                                        iters=self.global_step)
                
                self.writer.add_scalar('Loss/loss', loss.item(), (epoch-1)*self.args.train_batch*len(self.data_loader['train']) + self.args.train_batch*(idx))
                self.writer.add_scalar('Loss/loss_cls', loss_cls.item(), (epoch-1)*self.args.train_batch*len(self.data_loader['train']) + self.args.train_batch*(idx))
                self.writer.add_scalar('Loss/loss_rg', addition_loss.item(), (epoch-1)*self.args.train_batch*len(self.data_loader['train']) + self.args.train_batch*(idx))
                
                # print(i)
                if (idx) % (train_len // 5) == 0:
                    print('Epoch %d: train/loss %.4f, train/accr %.4f' % (self.global_step, loss.data.cpu(), acc.data.cpu()))

            # evaluation
            val_acc, h = self.eval(partition='val')

            is_best = 0

            if val_acc >= self.val_acc:
                self.val_acc = val_acc
                is_best = 1

            print('===>  Epoch %d: val/accr %.4f, val/best_accr %.4f' % (self.global_step, val_acc, self.val_acc))

            self.save_checkpoint({
                'iteration': self.global_step,
                'enc_module_state_dict': self.enc_module.state_dict(),
                'lp_module_state_dict': self.lp_module.state_dict(),
                'val_acc': val_acc,
                'optimizer': self.optimizer.state_dict(),
                }, is_best)

    def eval(self, partition='test', log_flag=True):
        best_acc = 0
        # set edge mask (to distinguish support and query edges)

        batch_size = self.args.test_batch
        num_supports = self.args.nKnovel * self.args.nExemplars
        num_queries = self.args.nTestNovel
        num_samples = num_supports + num_queries
        acc_all = []
        
        # set as eval mode
        self.enc_module.eval()
        self.lp_module.eval()

        # for each iteration
        with torch.no_grad():
            for i, (support_data, support_word, support_label, query_data, query_word_real, query_label) in enumerate(self.data_loader[partition]):
                # load task data list
                support_data = support_data.cuda()
                support_word = support_word.cuda()
                support_label = support_label.cuda()
                query_data = query_data.cuda()
                query_word_real = query_word_real.cuda()
                query_label = query_label.cuda()

                query_word = torch.zeros(query_data.size(0), query_data.size(1), support_word.size(2)).cuda()

                # set as single data
                full_data = torch.cat([support_data, query_data], 1)

                # (1) encode data
                full_data_embeddings_view = self.enc_module(full_data.view(-1, *full_data.shape[2:]))
                full_data_embeddings = full_data_embeddings_view.view(*full_data.shape[:2], -1)
                support_data_embeddings = full_data_embeddings[:, :num_supports]
                query_data_embeddings = full_data_embeddings[:, num_supports:]

                # (2) propagation
                support_data_embeddings_tiled = support_data_embeddings.unsqueeze(1).repeat(1, num_queries, 1, 1)
                support_data_embeddings_tiled = support_data_embeddings_tiled.view(batch_size*num_queries, num_supports, -1)
                query_data_embeddings_tiled = query_data_embeddings.contiguous().view(batch_size*num_queries, -1).unsqueeze(1)

                support_word_tiled = support_word.unsqueeze(1).repeat(1, num_queries, 1, 1)
                support_word_tiled = support_word_tiled.view(batch_size*num_queries, num_supports, -1)
                query_word_tiled = torch.zeros(batch_size*num_queries, 1, query_word.size(2)).cuda()

                query_word_real_tiled = query_word_real.contiguous().view(batch_size*num_queries, -1).unsqueeze(1)

                support_label_tiled = support_label.unsqueeze(1).repeat(1, num_queries, 1)
                support_label_tiled = support_label_tiled.view(batch_size*num_queries, num_supports)
                query_label_tiled = query_label.contiguous().view(batch_size*num_queries, -1)

                full_data_embeddings_tiled = torch.cat([support_data_embeddings_tiled, query_data_embeddings_tiled], dim=1)
                full_word_tiled = torch.cat([support_word_tiled, query_word_tiled], dim=1)
                full_word_real_tiled = torch.cat([support_word_tiled, query_word_real_tiled], dim=1)

                embeddings, addition_loss = self.lp_module(full_data_embeddings_tiled, full_word_tiled, full_word_real_tiled)

                # print(embeddings.size())
                support_embeddings = embeddings[:, :num_supports, :]
                query_embeddings = embeddings[:, -1, :].unsqueeze(1)

                prototype = get_prototypes(support_embeddings, support_label_tiled, self.args.nKnovel)
                acc = get_proto_accuracy(prototype, query_embeddings, query_label_tiled)

                acc_all.append(acc.data.cpu())

        acc_mean, h = self.mean_confidence_interval(acc_all)

        return acc_mean, h

    def adjust_learning_rate(self, optimizers, lr, iters):
        # new_lr = lr * (0.5 ** (int(iters / 15)))

        if iters in self.args.schedule:
            lr *= 0.1
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].cuda()

    def save_checkpoint(self, state, is_best):
        torch.save(state, os.path.join(self.args.save_dir, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(self.args.save_dir, 'checkpoint.pth.tar'),
                            os.path.join(self.args.save_dir, 'model_best.pth.tar'))

    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0*np.asarray(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        return m, h