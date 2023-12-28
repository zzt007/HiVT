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
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import ArgoverseV1DataModule
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser() # 创建命令行参数解析器对象
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args() # 解析命令行参数


    '''
    ModelCheckpoint主要有三个功能：检查指标、保存策略、模型状态保存
    ModelCheckpoint的工作原理是：监视训练过程中指定的指标，当指标达到设定的条件时，触发保存checkpoint的操作；
    可以设定保存checkpoint的策略，比如只保存最优模型；
    满足保存条件后，会将当前模型的权重、优化器状态等信息保存到磁盘上，在训练结束或者中断后，可以加载这些checkpoint来恢复模型，避免重新训练。
    '''
    # 下方ModelCheckpoint的参数意思为：监控指定monitor，保存save_top_k个模型，指定模式为最小化验证集损失
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    # 通过argparse解析的命令行参数创建一个训练器对象，并使用model_checkpoint回调来管理checkpoint的保存
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint])
    model = HiVT(**vars(args))
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    # 开始训练模型
    trainer.fit(model, datamodule)

# 学习在windows vscode下使用git进行代码版本管理 231228
# 同步失败，说fail to connect，不知道什么问题

# 使用git pull更新本地代码，现在用的是desktop，将此句更新上传至远程仓库，待会尝试在laptop上pull 231228-14：43
# 成功git pull更新，并且，当出现fail to connect时，打开梯子，并且手动设置一下，export HTTPS_PROXY