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
import os
# itertools是python标准库中的一个模块，提供一系列用于生成和处理迭代器的函数。
# permutations函数生成序列的所有可能的排列
from itertools import permutations
# product函数生成笛卡尔积
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData

# 这里继承的是geometric.data中的Dataset基类，简单来说就是用于创建图数据
# 详情参考：https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset
class ArgoverseV1Dataset(Dataset):
    # root:数据集存放的地址（目录）、split：按目的划分数据集、transform：对数据进行转换 ，转换可以理解为对数据的操作，如缩放scale之类？  
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        # 下划线前缀通常用于将属性与类名称区分，f代表将split变量插入到字符串中
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        # os.path.splitext函数将文件名分割为两部分，前部分是文件名（不包含扩展名），后部分是扩展名
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        # 将上一句得到的文件存入到processed_dir中，processed_dir的定义在下方有说明 = os.path.join(self.root,self._directory,'processed')
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    # @property是python的一个装饰器，用于将一个方法转为属性，属性是类的一个成员，可以直接通过类名访问，而不需要使用点（.）运算符
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    # Union是一个类型注释，用于指定一个类型的变量可以具有多个可能的类型。在此处，raw_file_names方法的返回值可以是str、List[str]、Tuple
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        am = ArgoverseMap()
        # 对每个文件进行处理
        for raw_path in tqdm(self.raw_paths):
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float) -> Dict:
    # 从raw_path中读取csv文件
    df = pd.read_csv(raw_path)

    # filter out actors that are unseen during the historical time steps，去除掉那些在观测时间内不可见的车辆
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    # isin是python中的一个内置方法，用于检查一个数组中的元素是否存在于另一个数组中
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    # 获取观测时间内对应的车辆ID信息
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    # 统计观测时间内车辆的数目
    num_nodes = len(actor_ids)

    # AV是argoverse数据集中对自车的标签
    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])
    # AGENT是argoverse数据集中对目标智能体的标签
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]

    # make the scene centered at AV，让场景以自车为中心
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
    # 自车的航向矢量
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
    # 航向角
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    # 使用自车航向角参数化的旋转矩阵
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    # initialization
    # 初始化x，num_nodes是车辆总数，50是观测时间+预测时间对应的点数（10hz），2是对应车辆xy坐标值
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    '''
    边的索引，permutation用于生成所有可能的边（在所有节点数之间两两组合），permutation方法返回一个包含所有可能边的列表，然后list将其转化为python的列表类型，然后torch再将其转为LongTensor张量；
    t().contiguous()方法将张量的形状转为（2，num_edges），其中num_edges是边的数量；
    具体形状可以试着自己打印一下。打印出来，发现这个边的表述其实类似于sparse COO format。belike👇 (这里是取了num_nodes=5)
    tensro([[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],
            [1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3]])
    '''
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    # 创建相关mask和旋转角的矩阵
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool) # 创建初始都是true
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    # 按照指定的列进行分组，这里按‘TRACK_ID’划分
    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True
        # 获取车辆的xy坐标信息
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        # xy-origin 即从绝对坐标转成相对坐标（相对自车的origin时刻，即观测时间末时刻），再与旋转矩阵相乘，得到以观测时间末时刻为中心的坐标系下的值
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately) 相当于对除自车以外的其它车的航向矢量和旋转角进行更新？
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True

    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]

    positions = x.clone()
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1: 20] - x[:, : 19])
    x[:, 0] = torch.zeros(num_nodes, 2)

    # get lane features at the current time step
    df_19 = df[df['TIMESTAMP'] == timestamps[19]]
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']]
    node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius)

    y = None if split == 'test' else x[:, 20:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]

    return {
        'x': x[:, : 20],  # [N, 20, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
