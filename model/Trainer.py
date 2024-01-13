import os
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm, trange
from model.SimGNN import SimGNN
from scipy.stats import spearmanr, kendalltau
from utils.utils import print_evals, prec_at_ks, calculate_ranking_correlation
from utils.utils import create_test_pairs_id, create_train_pairs_id, create_validate_pairs_id

class Trainer(object):
    def __init__(self, config, norm_ged):
        self._lr = config['lr']                             # 模型学习率
        self._norm_ged = norm_ged                           # 归一化的 GED，还不是 target
        self._wandb  = config['wandb']                      # 是否开启wandb记录模型损失
        self._epochs = config['epochs']                     # 模型训练代数
        self._log_path = config['log_path']                 # 日志存放路径
        self._patience = config['patience']                 # 早停的 patience 值
        self._batch_size = config['batch_size']             # 模型 batch_size
        self._start_val_iter = config['start_val_iter']     # 开始验证的代数
        self._every_val_iter = config['every_val_iter']     # 开始验证后，验证的间隔代数
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        
        self._model = SimGNN(config).to(self._device)

        if os.path.exists(config['log_path']) is False:
            os.makedirs(config['log_path'])
    
    def _validate(self, validaite_pairs_id, train_data):
        self._model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for (G1_id, G2_id) in tqdm(validaite_pairs_id, total=len(validaite_pairs_id), desc="Validate"):
                data = dict()
                data['g1'] = train_data[G1_id].to(self._device)
                data['g2'] = train_data[G2_id].to(self._device)
                targets.append( torch.exp(-self._norm_ged[G1_id][G2_id]).view(-1).to(self._device) )
                predictions.append( self._model(data) )
            vloss = F.mse_loss(torch.cat(predictions), torch.cat(targets))
        return vloss
    
    def fit(self, train_data):
        # 训练过程
        print("\n======= SimGNN training in {}. =======\n".format(self._log_path.split('/')[-2]))
        if self._wandb:
            wandb.init(
                project='SimGNN',
                name=self._log_path.split('/')[-1],
                config={
                    'learning_rate': self._lr,
                    'dataset': self._log_path.split('/')[-1],
                })

            # wandb.watch(self._model, log='all', log_graph=True, log_freq=10)

        cur_patience = 0
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr)
        train_pairs_id = create_train_pairs_id(len(train_data), self._batch_size)
        validaite_pairs_id = create_validate_pairs_id(len(train_data))

        epochs = trange(self._epochs, leave=True, desc="Epoch")
        min_vloss = 99999.0
        for epoch in epochs:
            self._model.train()
            cur_tloss = 0.0
            for batch_id in tqdm(train_pairs_id, total=len(train_pairs_id), desc="Train"):
                predictions, targets = [], []
                optimizer.zero_grad()
                for G1_id, G2_id in batch_id:
                    data = dict()
                    data['g1'] = train_data[G1_id].to(self._device)
                    data['g2'] = train_data[G2_id].to(self._device)
                    targets.append( torch.exp(-self._norm_ged[G1_id][G2_id]).view(-1).to(self._device) )
                    predictions.append( self._model(data) )
                loss = F.mse_loss(torch.cat(predictions), torch.cat(targets)) # 一整个batch的损失
                cur_tloss += loss.item()
                loss.backward()
                optimizer.step()
                
            cur_tloss = round(cur_tloss / len(train_pairs_id), 5)
            if self._wandb:
                wandb.log({'train_loss': cur_tloss})
            else:
                with open(self._log_path + 'train_loss.txt', 'a') as f:
                    f.write(str(epoch) + '\t' + str(cur_tloss) + '\n')
            epochs.set_description("Epoch (Loss=%g)" % cur_tloss)
            
            if epoch + 1 >= self._start_val_iter:
                if epoch % self._every_val_iter != 0:
                    continue
                torch.cuda.empty_cache()
                cur_vloss = self._validate(validaite_pairs_id, train_data).item()
                torch.cuda.empty_cache()

                cur_vloss = round(cur_vloss, 5)
                if min_vloss > cur_vloss:
                    min_vloss = cur_vloss
                    cur_patience = 0
                    self._save()
                else:
                    cur_patience += 1

                if self._wandb:
                    wandb.log({'valid_loss': cur_vloss, 'cur_patience': cur_patience})
                else:
                    with open(self._log_path + 'valid_loss.txt', 'a') as f:
                        f.write(str(epoch) + '\t' + str(cur_vloss) + '\t' + str(cur_patience) + '\n')
                if cur_patience >= self._patience:
                    print("Early Stop!")
                    break
        if self._wandb:
            wandb.finish()

    def score(self, train_data, test_data):
        # 测试过程
        print("\n======= SimGNN testing in {}. =======\n".format(self._log_path.split('/')[-2]))
        self._load()
        self._model.eval()
        
        scores = np.zeros( (len(test_data), len(train_data)) )
        ground_truth = np.zeros( (len(test_data), len(train_data)) )
        prediction_mat = np.zeros( (len(test_data), len(train_data)) )
        rho_list = []
        tau_list = []
        prec_at_10_list = []
        prec_at_20_list = []

        test_pairs_id = create_test_pairs_id(len(train_data), len(test_data))
        
        with torch.no_grad():
            for (G1_id, G2_id) in tqdm(test_pairs_id, total=len(test_pairs_id), desc="Test"):
                data = dict()
                data['g1'] = train_data[G1_id].to(self._device)
                data['g2'] = test_data[G2_id].to(self._device)

                pred = self._model(data).cpu()
                targ = torch.exp(-self._norm_ged[G1_id][G2_id + len(train_data)])
                ground_truth[G2_id][G1_id] = targ.cpu().numpy()
                prediction_mat[G2_id][G1_id] = pred.cpu().numpy()
                scores[G2_id][G1_id] = F.mse_loss(pred, targ.view(-1)).detach().cpu().numpy()
            for i in range(len(test_data)):
                rho_list.append( calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]) )
                tau_list.append( calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]) )
                prec_at_10_list.append( prec_at_ks(ground_truth[i], prediction_mat[i], 10) )
                prec_at_20_list.append( prec_at_ks(ground_truth[i], prediction_mat[i], 20) )

        print_evals(np.mean(scores), np.mean(rho_list), np.mean(tau_list), np.mean(prec_at_10_list), np.mean(prec_at_20_list))

    def _save(self):
        torch.save(self._model.state_dict(), self._log_path + 'best_model.pt')

    def _load(self):
        self._model.load_state_dict(torch.load(self._log_path + 'best_model.pt'))
