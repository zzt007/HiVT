# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import DistanceDropEdge
from utils import TemporalData
from utils import init_weights

# 这里的类为什么又是直接继承nn.Module，而不是像之前一样继承pl.LightningModule?
class LocalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False) -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        # parallel 是指避免for循环，利用tensor并行计算
        self.parallel = parallel

        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel)
        self.temporal_encoder = TemporalEncoder(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
        self.al_encoder = ALEncoder(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)
    #  -> 这个符号代表指定方法的返回值类型，这里是torch.Tensor
    def forward(self, data: TemporalData) -> torch.Tensor:
        for t in range(self.historical_steps): # 沿时间轴，在每一时刻上进行
            # 获取当前时刻的子图
            # 变量后的 _ 表示忽略该值，由于subgraph函数接收data.edge_index这个二元组作为输入，所以返回也是二元组，但是等号左侧赋值变量只有一个，另一个使用 _ 表示忽略
            data[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index)
            # \ 出现在行末尾表示 续行符，下一行接着
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]
        if self.parallel:
            # 如果使用并行计算，则创建包含historical_steps个数据的空列表，然后遍历每个historical_step
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                # drop_edge函数用于去除边，self.drop_edge = DistanceDropEdge(local_radius),这个对象返回的是edge_index和edge_attr（详情见utils.py中）
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                # snapshots列表中的每个元素都是torch_geometric.data类型的(这里表示为Data),该数据类型详情可参见:https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
                # 续上一行,总的来说,snapshots列表中的每个元素包含了对应时刻的节点特征x,边索引(coo format),边特征,图的节点数量 这些信息
                snapshots[t] = Data(x=data.x[:, t], edge_index=edge_index, edge_attr=edge_attr,
                                    num_nodes=data.num_nodes)
            # 将snapshots聊表转换为batch对象,并使用aa_encoder对输入进行编码
            batch = Batch.from_data_list(snapshots)
            # 输出的形状被reshape成[T,N,D],其中T为时间步数,N为节点数目,D为编码结果的维度
            out = self.aa_encoder(x=batch.x, t=None, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                                  bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
            out = out.view(self.historical_steps, out.shape[0] // self.historical_steps, -1)
        else:
            # 如果不采用并行计算,则遍历每个时间步,后续操作和上面一样,得到A-A interaction的out
            out = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                out[t] = self.aa_encoder(x=data.x[:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                                         bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat'])
            out = torch.stack(out)  # [T, N, D]
        # 将A-A interaction的输出作为temporal_encoder的输入,经过temporal_encoder后输入到下方的A-L interaction中
        out = self.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, : self.historical_steps])
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
                              is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],
                              traffic_controls=data['traffic_controls'], rotate_mat=data['rotate_mat'])
        # 至此,完成local_encoder的所有内容,得到的输出out准备进入global_encoder
        return out

# AAEncoder类继承自MessagePassing,是图神经网络传递信息的基类,可参考:https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#the-messagepassing-base-class
class AAEncoder(MessagePassing):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,
                 **kwargs) -> None:
        # 调用torch_geometric.nn.MessagePassing类的构造函数,并将aggr参数设置为'add', aggr是aggregate聚合,用于消息传递的聚合方式,还可以设置为'mean\max'之类的
        # debug时发现其flow是：source_to_target
        super(AAEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel
        # 中心智能体的embedding，（2，64）
        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        # 邻近智能体的embedding,in_channels:(2,2),out_channels:64
        self.nbr_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        # 这个是和attention drop技术一样的吗？
        self.attn_drop = nn.Dropout(dropout)
        # 下面这两个ih 和 hh 是什么的简称，现在还不知道。 231229--16：52
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # 映射、投射的dropout，其实就是维度的变化
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        # bos_token表示BOS(开始)符号的嵌入表示,EOS是结束符号.在序列信息处理中,BOS表示Begining Of Sequence, EOS表示End Of Sequence
        # shape:torch.size([20,64])
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        # init.normal_是按照正态分布为bos_token进行赋值，该正态分布的均值为0，标准差为0.02，生成服从该分布的随机数，或许有利于模型的训练和收敛
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                t: Optional[int],
                edge_index: Adj, # Adj 表示一个邻接矩阵（adjacency matrix）
                edge_attr: torch.Tensor,
                bos_mask: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        if self.parallel:
            if rotate_mat is None:
                # self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)，return embed(x)
                center_embed = self.center_embed(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1))
            else:
                # 将rotate_mat扩展到和x相同的形状，expand方法不会修改原始张量的形状，而是创建一个新的扩展后的张量。 *sizes，sizes是可变参数，表示要扩展的新形状，若维度为-1，则表示不变。
                # unsqueeze()和squeeze()都是用于操作张量的维度，前者用于在指定维度上插入一个新的维度，后者用于删除张量中大小为1的维度
                center_embed = self.center_embed(
                    torch.matmul(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1).unsqueeze(-2),
                                 rotate_mat.expand(self.historical_steps, *rotate_mat.shape)).squeeze(-2))
            '''
            torch.where是用于根据给定条件选择张量的不同部分，第一个参数是condition（布尔张量），用于指定要在x或y中选择的条件。
            x和y是需要进行比较的两个张量，如果condition的某个元素为true,则返回与x相同位置的元素，如果condition的某个元素为false，则返回与y相同位置的元素。
            '''
            center_embed = torch.where(bos_mask.t().unsqueeze(-1),
                                       self.bos_token.unsqueeze(-2),
                                       center_embed).view(x.shape[0], -1)
        else:
            if rotate_mat is None:
                center_embed = self.center_embed(x)
            else:
                center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2))
            center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)
        # 这里用的残差网络的思想，本身 加上 经过mha处理的。这里要看标准transformer的一个流程图，就可以清晰地知道。
        center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat,
                                                      size)
        # ff block是前馈神经网络模块，经过上面一句的残差concatenate后，输入进ff中，又有一次残差
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

    # 负责计算每条边的信息 
    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            # 对邻近智能体进行embedding，原句为self.nbr_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
            # 下面这句只输入了第一个in_channels参数（list类型的），第二个参数没有指明，这样调用不会报type error吗？
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            if self.parallel:
                # repeat（x,times,axis），x代表输入的矩阵，times代表每个元素需要重复的次数，axis代表指定在哪一维度上重复
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                # 得到基于中心智能体的旋转矩阵，见论文中有描述
                center_rotate_mat = rotate_mat[edge_index[1]]
            # 利用nbr_embed，即MultipleInputEmbedding对带有旋转矩阵的输入进行编码，得到邻近智能体的embedding
            nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                        torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
        # 中心智能体的embedding作为q，邻近智能体的embedding作为k和v，进行qkv计算
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    # 负责更新每个节点的特征
    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        # 根据论文所述的gating function来进行更新，在论文中，g=sigmoid（W^gate[z,m])，其中z为中心智能体的embedding，m为 alpha*value（environment features）
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        # 这里其实就是对应论文中的公式（7），写成这样或许更容易看懂： gate*(self.lin_self(center_embed)) + (1-gate)*inputs 。也就是说，这里inputs对应m（environment features）
        return inputs + gate * (self.lin_self(center_embed) - inputs)
    
    # 多头注意力模块
    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        # 原型为self.out_proj = nn.Linear(embed_dim, embed_dim)，但这里采用self.propagate方法的返回值作为out_proj方法的参数
        '''
        对self.propagate方法的说明：在整个文件中，没有显式地定义这个方法，但是为什么能调用呢？这是因为AAEncoder类是继承自MessagePassing基类的，所以能够调用父类的方法。
        值得一提的是，propagate方法在内部还会调用message() , aggregate() , update()方法，这也就是为什么在整段代码中都没有看到哪里有显式地调用这几个方法。
        具体可以参考官方文档：https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        '''
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                    edge_attr=edge_attr, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(center_embed)
    
    # 前馈神经网络，这里由一个mlp实现，mlp的具体结构见上面的初始化init定义
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        # nn.Parameter用于表示模型中的可学习参数。
        # padding_token，一般输入是固定长度的序列，所以要对不满足固定长度的序列进行padding; shape = torch.size([20,1,64])
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        # 在序列数据的开头添加cls_token，可以参考BERT中的做法,是bert中用于区分token属于哪一个序列的做法之一;shape = torch.size([1,1,64])
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        # positional embedding，用于获取序列的时序信息; shape = torch.size([21,1,64]),这里的21是因为论文中所述的在序列末尾append了一个可学习的token吗？还是因为在nlp中处理一般都是把第一行空出，所以才要加？
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        # attention mask，用于在训练过程中为每个样本屏蔽掉当前时刻之后的数据
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        # 开辟attention mask的缓冲区，使其可以被模型访问和使用，在模型的前向传播过程中，可以直接使用mask而不用每次都重新计算；且当其被register成模型的属性后，可以在训练过程中被优化器更新
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        # 这里主要是给出关于输入x的representation，同样也是参考BERT中的做法，对于给定的输入token，其representation可以由三部分组成，即 token本身的embedding + cls_token(表示该token在哪个序列中) + positional embedding
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        # 得到x的representation后，输入进nn.TransformerEncoder方法中进行编码
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    # 在transformer中常见的生成attention mask的写法，网上有很多可以参考；此处seq_len = 21
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        # 直接调用nn.Module里带的多头注意力机制实现
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # attn_mask,key_paddingg_mask,need_weights这些是MultiheadAttention的其他可选参数，具体可见nn的参考文档
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)

# 感觉涉及到interaction的类都是基于torch_geometric实现的，涉及到transformer相关的是基于nn.Module。这里是A-L interaction，所以继承自基类MessagePassing
class ALEncoder(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        # 同样如A-A interaction中的，指定消息聚合(aggregate)方式为add
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
        nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                is_intersections: torch.Tensor,
                turn_directions: torch.Tensor,
                traffic_controls: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        x_lane, x_actor = x
        is_intersections = is_intersections.long()
        turn_directions = turn_directions.long()
        traffic_controls = traffic_controls.long()
        '''
        out = self.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, : self.historical_steps])
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
                              is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],
                              traffic_controls=data['traffic_controls'], rotate_mat=data['rotate_mat'])
        上述三句代码表示调用al_encoder的具体过程，可以看到其输入x=(data['lane_vectors'],out) ,lane_vectors是知道的，其中out是经过temporal_encoder后的编码表示
        '''
        # 这里的x_actor = out，即是中心智能体经过temporal encoder后的编码表示。然后再与 x_lane(即lane_vectors)在多头注意力机制中进行交互，最终得到A-L interaction结果
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections,
                                            turn_directions, traffic_controls, rotate_mat, size)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        # 至此，完成了local_encoder的所有内容，得到了可以输入进global_interaction的变量x_actor
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                is_intersections_j,
                turn_directions_j,
                traffic_controls_j,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            # self.lane_embed即为 MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
            x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                                   torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        # x_i其实就是中心智能体i，x_j就是邻近智能体j，根据论文所述，i为query，j为key和value，然后进行qkv运算
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    # 和之前A-A interaciton中update差不多
    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   is_intersections: torch.Tensor,
                   turn_directions: torch.Tensor,
                   traffic_controls: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                               is_intersections=is_intersections, turn_directions=turn_directions,
                                               traffic_controls=traffic_controls, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)
