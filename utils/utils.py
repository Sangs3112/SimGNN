import yaml
import torch
import random
import numpy as np

from texttable import Texttable
from torch_geometric.utils import degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

def print_evals(mse_error, rho, tau, p10, p20):
    r"""
        用于模型test阶段结束后，打印所有的结果
        参数 mse_error: float，模型预测的MSE
        参数 rho: 斯皮尔曼相关系数
        参数 tau: 肯德尔相关系数
        参数 p10: p@10
        参数 p20: p@20
    """
    print("mse(10^-3): " + str(round(mse_error * 1000, 5)) + '.')
    print("rho: " + str(round(rho, 5)) + '.')
    print("tau: " + str(round(tau, 5)) + '.')
    print("p@10: " + str(round(p10, 5)) + '.')
    print("p@20: " + str(round(p20, 5)) + '.')

def calculate_ranking_correlation(rank_corr_function, prediction, target):
    r"""
        计算相关系数
        参数 rank_corr_function: 函数，是scipy.stats中的函数
        参数 prediction: double，是模型的预测值
        参数 target: double，是数据集的准确值
        返回 相关系数
    """
    def ranking_func(data):
        sort_id_mat = np.argsort(-data)
        n = sort_id_mat.shape[0]
        rank = np.zeros(n)
        for i in range(n):
            finds = np.where(sort_id_mat == i)
            fid = finds[0][0]
            while fid > 0:
                cid = sort_id_mat[fid]
                pid = sort_id_mat[fid - 1]
                if data[pid] == data[cid]:
                    fid -= 1
                else:
                    break
            rank[i] = fid + 1
        return rank
    
    r_prediction = ranking_func(prediction)
    r_target = ranking_func(target)

    return rank_corr_function(r_prediction, r_target).correlation

def prec_at_ks(true_r, pred_r, ks, rm=0):
    r"""
        计算 p@k
        参数 true_r: double，数据集的真实值
        参数 pred_r: double，模型的预测值
        参数 ks: int，k的值
        返回 ps: int，最后的p@k
    """
    def top_k_ids(data, k, inclusive, rm):
        """
        :param data: input
        :param k:
        :param inclusive: whether to be tie inclusive or not.
            For example, the ranking may look like this:
            7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
            If tie inclusive, the top 1 results are [7, 9].
            Therefore, the number of returned results may be larger than k.
            In summary,
                len(rtn) == k if not tie inclusive;
                len(rtn) >= k if tie inclusive.
        :param rm: 0
        :return: for a query, the ids of the top k database graph
        ranked by this model.
        """
        sort_id_mat = np.argsort(-data)
        n = sort_id_mat.shape[0]
        if k < 0 or k >= n:
            raise RuntimeError('Invalid k {}'.format(k))
        if not inclusive:
            return sort_id_mat[:k]
        # Tie inclusive.
        dist_sim_mat = data
        while k < n:
            cid = sort_id_mat[k - 1]
            nid = sort_id_mat[k]
            if abs(dist_sim_mat[cid] - dist_sim_mat[nid]) <= rm:
                k += 1
            else:
                break
        return sort_id_mat[:k]
    true_ids = top_k_ids(true_r, ks, inclusive=True, rm=rm)
    pred_ids = top_k_ids(pred_r, ks, inclusive=True, rm=rm)
    ps = min( len(set(true_ids).intersection(set(pred_ids)) ), ks) / ks
    return ps

def create_training_batches_all(training_len, batch_size):
    r"""
        生成训练的batch，将来需要在每个epoch中都进行调用
        参数 training_len: int, 训练集的长度，则图id范围为[0, training_len - 1]
        参数 batch_size: int, batch的大小
        返回 batches: 双层list，除了最后一个以外，其他内部list长度均为batch_size。
    """
    train_graph_list = list(range(training_len))
    combinations = [(i, j) for i in train_graph_list for j in train_graph_list if i < j]
    random.shuffle(combinations)
    batches = [combinations[i:i + batch_size] for i in range(0, len(combinations), batch_size)]
    return batches

def create_train_pairs_id(training_len, batch_size):
    real_training_len = int(training_len * 0.75)
    return create_training_batches_all(real_training_len, batch_size)

def create_validate_pairs_id(training_len):
    real_training_len = int(training_len * 0.75)
    return [(i, j) for i in range(real_training_len) for j in range(real_training_len, training_len)]

def create_test_pairs_id(training_len, testing_len):
    real_training_len = int(training_len * 0.75)
    return [(i, j) for i in range(real_training_len) for j in range(testing_len)]

def nice_printer(config):
    r"""
        打印配置
        参数 config: 字典，键值分别为参数的名称和对应的参数值
    """
    tabel_data = [['Key', 'Value']] + [[k, v] for k, v in config.items()]
    t = Texttable().set_precision(4)
    t.add_rows(tabel_data)
    print(t.draw())

def set_seed(seed):
    r"""
        设置所有的随机数种子都为seed
        参数 seed: int， 将要设置的随机数种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 启用确定性计算，降低性能，但是确保实验可重复，尤其是dropout等随机性操作
    torch.backends.cudnn.benchmark = False      # 禁用自动调整策略，提高稳定性

def get_config(args):
    r"""
        通过传入的args参数，得到完整的 config字典
        参数 args: 默认包含一些main函数中公用参数
        返回 config: 完整参数字典，键值分别为参数名和参数对应的值
    """

    config = _get_part_config('utils/config.yml')['SimGNN']
    config.update( _get_part_config('utils/config.yml')[args.dataset] )
    config['log_path'] = args.log_path + args.dataset + '/'

    return config

def _get_part_config(config_path):
    r"""
        读取config.yml文件
        参数 config_path: config.yml文件路径
        返回 config: 字典，键值分别为参数名和参数对应的值
    """
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def load_data(path, dataset_name):
    path = path + dataset_name
    train_data = GEDDataset(path, dataset_name)
    test_data = GEDDataset(path, dataset_name, False)
    norm_ged = train_data.norm_ged

    # 但是LINUX，IMDBMuliti，ALKANE这三个数据集的并没有节点的类型x
    if train_data[0].x is None:
        train_data, test_data = _process_feature(train_data, test_data)

    return train_data, test_data, norm_ged

def _process_feature(train_data, test_data):
    max_degree = 0
    for g in (train_data + test_data):
        if g.edge_index.size(1) > 0:
            max_degree = max( max_degree, int(degree(g.edge_index[0]).max().item()) )
    one_hot_degree = OneHotDegree(max_degree, cat=False)
    train_data.transform = one_hot_degree
    test_data.transform = one_hot_degree
    return train_data, test_data
