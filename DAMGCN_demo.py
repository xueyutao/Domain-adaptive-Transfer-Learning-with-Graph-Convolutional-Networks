# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData, DomainData2
from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='card2')
parser.add_argument("--target", type=str, default='card')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int, default=21)
parser.add_argument("--UDAGCN", type=bool, default=True)
parser.add_argument("--encoder_dim", type=int, default=16)


args = parser.parse_args()
seed = args.seed
use_UDAGCN = args.UDAGCN
encoder_dim = args.encoder_dim



id = "source: {}, target: {}, seed: {}, UDAGCN: {}, encoder_dim: {}"\
    .format(args.source, args.target, seed, use_UDAGCN,  encoder_dim)

print(id)



rate = 0.0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
dataset = DomainData2("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
print(source_data)
print(source_data.x)
print(source_data.y)
print(source_data.edge_index)
print(source_data.train_mask)
print(dataset.num_features)
print(dataset.num_classes)
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
print(target_data)
print(target_data.x.shape)
print(target_data.y.shape)
print(target_data.edge_index)
print(target_data.train_mask)
print(dataset.num_classes)


source_data = source_data.to(device)
target_data = target_data.to(device)



# 通用GNN，共享权重
class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]


        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv

        self.conv_layers = nn.ModuleList([
            model_cls(dataset.num_features, 128,
                     weight=weights[0],
                     bias=biases[0],
                      **kwargs),
            model_cls(128, encoder_dim,
                     weight=weights[1],
                     bias=biases[1],
                      **kwargs)
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x

# 梯度反转，域适应
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


loss_func = nn.CrossEntropyLoss().to(device)

# 编码器
encoder = GNN(type="gcn").to(device)

if use_UDAGCN:
    # ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=10).to(device)
    ppmi_encoder = GNN(type="gcn").to(device)
    # ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=2).to(device)


# 解码器
cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)

domain_model = nn.Sequential(
    GRL(),
    nn.Linear(encoder_dim, 40),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(40, 2),
).to(device)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


# att_model = Attention(encoder_dim).cuda()
att_model = Attention(encoder_dim)

models = [encoder, cls_model, domain_model]
if use_UDAGCN:
    # models.extend([ppmi_encoder, att_model])
    models.extend([ppmi_encoder])
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=3e-3)
print(models)

def gcn_encode(data, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def ppmi_encode(data, cache_name, mask=None):
    encoded_output = ppmi_encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def encode(data, cache_name, mask=None):
    # 只要测试的时候有mask
    gcn_output = gcn_encode(data, cache_name, mask)
    if use_UDAGCN:
        ppmi_output = ppmi_encode(data, cache_name, mask)
        # outputs = att_model([gcn_output, ppmi_output])
        # return outputs
        return ppmi_output
    else:
        return gcn_output

def predict(data, cache_name, mask=None):
    if cache_name=='source':
        encoded_output = gcn_encode(data, cache_name, mask)
    else:
        encoded_output = ppmi_encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    # accuracy = correct / len(labels)
    return accuracy


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    # 线性层结果，如[0.23,-0.34]
    logits = predict(data, cache_name, mask)
    # 结果是[0, 0 ,1]
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    from sklearn.metrics import roc_auc_score, f1_score
    auc_score = roc_auc_score(labels, F.softmax(logits, dim=-1)[:, 1].detach(), average='macro')
    macro_F = f1_score(labels, torch.argmax(logits, dim=-1), average='macro')
    print(' auc-roc: {:f}, macro_F: {:f}'.format(auc_score, macro_F))
    # from sklearn import metrics
    # macro_f1 = metrics.f1_score(labels, preds, average='macro')
    # precision = metrics.precision_score(labels, preds, average='macro')
    # recall = metrics.recall_score(labels, preds, average='macro')
    # print(macro_f1)
    # print(precision)
    # print(recall)
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(labels.data.cpu().numpy(), preds.data.cpu().numpy()))
    print(confusion_matrix(labels.data.cpu().numpy(), preds.data.cpu().numpy(), labels=[0, 1]))
    return accuracy



epochs = 300
def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, 0.05)

    # print(source_data)
    encoded_source = gcn_encode(source_data, "source")
    # es = encoded_source[source_data.train_mask]
    # print('1', encoded_source)
    encoded_target = ppmi_encode(target_data, "target")
    # print('2', encoded_target)
    source_logits = cls_model(encoded_source)

    # use source classifier loss:   源域分类
    # cls_loss = loss_func(source_logits, source_data.y)
    cls_loss = loss_func(source_logits, source_data.y)

    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3

    if use_UDAGCN:
        # use domain classifier loss:   对抗学习
        source_domain_preds = domain_model(encoded_source)
        target_domain_preds = domain_model(encoded_target)

        source_domain_cls_loss = loss_func(
            source_domain_preds,
            torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        target_domain_cls_loss = loss_func(
            target_domain_preds,
            # torch.zeros(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
            torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        loss_grl = source_domain_cls_loss + target_domain_cls_loss
        loss = cls_loss + loss_grl

        # use target classifier loss:  目标域分类
        encoded_target = encoded_target[target_data.train_mask]
        target_logits = cls_model(encoded_target)
        loss_entropy = loss_func(target_logits, target_data.y[target_data.train_mask])
        # target_probs = F.softmax(target_logits, dim=-1)
        # target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

        # loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        # loss = loss + loss_entropy * (epoch / epochs * 0.01)
        loss = loss + loss_entropy


    else:
        loss = cls_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
for epoch in range(1, epochs):
    train(epoch)
    source_correct = test(source_data, "source", source_data.test_mask)
    target_correct = test(target_data, "target", target_data.test_mask)
    print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch, source_correct, target_correct))
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_epoch = epoch
print("=============================================================")
line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}"\
    .format(id, best_epoch, best_source_acc, best_target_acc)

print(line)


