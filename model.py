import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from thop import profile

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path


def conv_block(in_channels,out_channels,use_relu=True):
    if use_relu:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )



def conv_block_mobile(in_channels, out_channels, use_relu=True):
    if use_relu:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )


class AvgBlock(BasicModule):
    def __init__(self, nFeat):
        super(AvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        labels_train_transposed = labels_train.transpose(1, 2)
        weight_novel = torch.bmm(labels_train_transposed, features_train)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))
        return weight_novel


class ConvNet_mobile(BasicModule):
    def __init__(self):
        super(ConvNet_mobile, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1, 64),
            conv_block_mobile(64, 64),
            conv_block_mobile(64, 128),
            conv_block_mobile(128, 128, use_relu=False),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        return out


class AttentionBlock(BasicModule):
    def __init__(self, nFeat, nKall, scale_att=10.0):
        super(AttentionBlock, self).__init__()

        self.nFeat = nFeat
        self.queryLayer = nn.Linear(nFeat, nFeat)
        self.queryLayer.weight.data.copy_(
            torch.eye(nFeat, nFeat) + torch.randn(nFeat, nFeat) * 0.001)
        self.queryLayer.bias.data.zero_()

        self.scale_att = nn.Parameter(torch.FloatTensor(1).fill_(scale_att), requires_grad=True)
        wkeys = torch.FloatTensor(nKall, nFeat).normal_(0.0, np.sqrt(2.0 / nFeat))
        self.wkeys = nn.Parameter(wkeys, requires_grad=True)
        # print('queryLayer', self.queryLayer.size())
        # print('scale_att', self.scale_att.size())
        # print('wkeys', self.wkeys.size())

    def forward(self, features_train, labels_train, weight_base, Kbase):
        batch_size, num_train_examples, num_features = features_train.size()
        nKbase = weight_base.size(1)  # [batch_size,nKbase,num_features]
        labels_train_transposed = labels_train.transpose(1, 2)
        nKnovel = labels_train_transposed.size(1)  # [batch_size,nKnovel,num_train_examples]

        features_train = features_train.view(batch_size * num_train_examples, num_features)
        Qe = self.queryLayer(features_train)
        Qe = Qe.view(batch_size, num_train_examples, self.nFeat)
        Qe = F.normalize(Qe, p=2, dim=Qe.dim() - 1, eps=1e-12)

        wkeys = self.wkeys[Kbase.view(-1)]
        wkeys = F.normalize(wkeys, p=2, dim=wkeys.dim() - 1, eps=1e-12)
        # Transpose from[batch_size,nKbase,nFeat]->[batch_size,nFeat,nKbase]
        wkeys = wkeys.view(batch_size, nKbase, self.nFeat).transpose(1, 2)

        # Compute the attention coefficients
        # AttenCoffiencients=Qe*wkeys ->
        # [batch_size x num_train_examples x nKbase] =[batch_size x num_train_examples x nFeat] * [batch_size x nFeat x nKbase]
        AttentionCoef = self.scale_att * torch.bmm(Qe, wkeys)
        AttentionCoef = F.softmax(AttentionCoef.view(batch_size * num_train_examples, nKbase))
        AttentionCoef = AttentionCoef.view(batch_size, num_train_examples, nKbase)

        # Compute the weight_novel
        # weight_novel=AttentionCoef * weight_base ->
        # [batch_size x num_train_examples x num_features] =[batch_size x num_train_examples x nKbase] * [batch_size x nKbase x num_features]
        weight_novel = torch.bmm(AttentionCoef, weight_base)
        # weight_novel=labels_train_transposed*weight_novel ->
        # [batch_size x nKnovel x num_features] = [batch_size x nKnovel x num_train_examples] * [batch_size x num_train_examples x num_features]
        weight_novel = torch.bmm(labels_train_transposed, weight_novel)
        # div K-shot ,get avg
        weight_novel = weight_novel.div(labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))
        return weight_novel


class LinearDiag(BasicModule):
    def __init__(self, num_features, bias=False):
        super(LinearDiag, self).__init__()
        weight = torch.FloatTensor(num_features).fill_(1)  # initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)

        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        assert (X.dim() == 2 and X.size(1) == self.weight.size(0))
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)

        return out


class Classifier(BasicModule):
    def __init__(self, nKall=7, nFeat=128 * 5 * 5, weight_generator_type='none'):  # v1:128*5*5 v2:1152
        super(Classifier, self).__init__()
        self.nKall = nKall
        self.nFeat = nFeat
        self.weight_generator_type = weight_generator_type

        weight_base = torch.FloatTensor(nKall, nFeat).normal_(0.0, np.sqrt(2.0 / nFeat))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        scale_cls = 10.0
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)
        # print('weight_base', weight_base.size())
        # print('bias', self.bias.size())
        # print('scale_cls', self.scale_cls.size())

        if self.weight_generator_type == 'none':
            # if type is none , then feature averaging is being used.
            # However,in this case the generator doesn't involve any learnable params ,thus doesn't require training
            self.favgblock = AvgBlock(nFeat)
        elif self.weight_generator_type == 'attention_based':
            scale_att = 10.0
            self.favgblock = AvgBlock(nFeat)
            self.attentionBlock = AttentionBlock(nFeat, nKall, scale_att=scale_att)

            self.wnLayerFavg = LinearDiag(nFeat)
            self.wnLayerWatt = LinearDiag(nFeat)
        else:
            raise ValueError('weight_generator_type is not supported!')

    def get_classification_weights(
            self, Kbase_ids, features_train=None, labels_train=None):
        """
        Args:
            Get the classification weights of the base and novel categories.
            Kbase_ids:[batch_size , nKbase],the indices of base categories that used
            features_train:[batch_size,num_train_examples(way*shot),nFeat]
            labels_train :[batch_size,num_train_examples,nKnovel(way)] one-hot of features_train

        return:
            cls_weights:[batch_size,nK,nFeat]
        """
        # get the classification weights for the base categories
        batch_size, nKbase = Kbase_ids.size()
        weight_base = self.weight_base[Kbase_ids.view(-1)]
        weight_base = weight_base.view(batch_size, nKbase, -1)

        # if training data for novel categories are not provided,return only base_weight
        if features_train is None or labels_train is None:
            return weight_base

        # get classification weights for novel categories
        _, num_train_examples, num_channels = features_train.size()
        nKnovel = labels_train.size(2)

        # before do cosine similarity ,do L2 normalize
        features_train = F.normalize(features_train, p=2, dim=features_train.dim() - 1, eps=1e-12)
        if self.weight_generator_type == 'none':
            weight_novel = self.favgblock(features_train, labels_train)
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type == 'attention_based':
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_novel_avg = self.wnLayerFavg(weight_novel_avg.view(batch_size * nKnovel, num_channels))

            # do L2 for weighr_base
            weight_base_tmp = F.normalize(weight_base, p=2, dim=weight_base.dim() - 1, eps=1e-12)

            weight_novel_att = self.attentionBlock(features_train, labels_train, weight_base_tmp, Kbase_ids)
            weight_novel_att = self.wnLayerWatt(weight_novel_att.view(batch_size * nKnovel, num_channels))

            weight_novel = weight_novel_avg + weight_novel_att
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        else:
            raise ValueError('weight generator type is not supported!')

        # Concatenate the base and novel classification weights and return
        weight_both = torch.cat([weight_base, weight_novel], dim=1)  # [batch_size ,nKbase+nKnovel , num_channel]

        return weight_both

    def apply_classification_weights(self, features, cls_weights):
        """
        Apply the classification weight vectors to the feature vectors
        Args:
            features:[batch_size,num_test_examples,num_channels]
            cls_weights:[batch_size,nK,num_channels]
        Return:
            cls_scores:[batch_size,num_test_examples(query set),nK]
        """
        # do L2 normalize
        features = F.normalize(features, p=2, dim=features.dim() - 1, eps=1e-12)
        cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim() - 1, eps=1e-12)
        cls_scores = self.scale_cls * torch.baddbmm(1.0,
                                                    self.bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1, 2))
        return cls_scores

    def forward(self, features_test, Kbase_ids, features_train=None, labels_train=None):
        """
        Recognize on the test examples both base and novel categories.
        Args:
            features_test:[batch_size,num_test_examples(query set),num_channels]
            Kbase_ids:[batch_size,nKbase] , the indices of base categories that are being used.
            features_train:[batch_size,num_train_examples,num_channels]
            labels_train:[batch_size,num_train_examples,nKnovel]

        Return:
            cls_score:[batch_size,num_test_examples,nKbase+nKnovel]

        """
        cls_weights = self.get_classification_weights(
            Kbase_ids, features_train, labels_train)
        cls_scores = self.apply_classification_weights(features_test, cls_weights)
        return cls_scores


if __name__ == '__main__':
    x = torch.randn(2, 84, 84)
    x_int = torch.randn(1, 1, 84, 84)
    net = ConvNet_mobile()
    print(net)
    out = net(x_int)
    print(out.size())
    n_parameters = sum([np.prod(p.size()) for p in net.parameters()])
    flops, params = profile(net, inputs=(x_int,))
    print('flops: ', flops)
    print('params: ', params)
    print('\nTotal number of parameters:', n_parameters)
    # print(net.shape)
    k_id = torch.randn(1, 7).long()
    print(k_id.size())
    classifier = Classifier(weight_generator_type='attention_based')
    print(classifier)
    out_c = classifier(out.view(1, 1, -1), k_id)
    print(out_c.size())

    n_parameters_c = sum([np.prod(m.size()) for m in classifier.parameters()])
    flop, param = profile(classifier, inputs=(out.view(1, 1, -1), k_id))
    print('flops: ', flop)
    # rint('params: ', param)
    print('\nTotal number of parameters:', n_parameters_c)