# PROTOCOL.md — 实验协议细节

## 协议 ID

`hod-gnn-replication-v1`

## 实验顺序

1. Paper-seeds-provisional: 使用临时种子 [0,1,2,3] 运行
2. Official-seeds: 使用各方法官方 seed 口径运行

## 方法运行命令

### GPS

```bash
# paper-seeds-provisional
conda activate graphgps
cd code/vendor/GraphGPS
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml seed=0 wandb.use=False
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml seed=1 wandb.use=False
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml seed=2 wandb.use=False
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml seed=3 wandb.use=False

# official-seeds
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml --repeat 10 wandb.use=False
```

### GraphViT

```bash
# paper-seeds-provisional
conda activate graph_mlpmixer
cd code/vendor/GraphViT
python -m train.zinc cfg.seed=0
python -m train.zinc cfg.seed=1
python -m train.zinc cfg.seed=2
python -m train.zinc cfg.seed=3
```

### Full / Random / Policy-Learn

```bash
# paper-seeds-provisional
cd code/vendor/policy-learn
python train.py dataset=zinc selection_type=all seed=1
python train.py dataset=zinc selection_type=all seed=2
python train.py dataset=zinc selection_type=all seed=3
python train.py dataset=zinc selection_type=all seed=4
```

## 数据集处理

| 数据集 | 加载方式 | split 来源 |
|--------|----------|------------|
| ZINC-12K | PyG ZINC(subset=False) | 固定预定义 |
| MOLTOX21 | OGB PygGraphPropPredDataset('ogbg-moltox21') | scaffold |
| MOLBACE | OGB PygGraphPropPredDataset('ogbg-molbace') | scaffold |
| MOLHIV | OGB PygGraphPropPredDataset('ogbg-molhiv') | scaffold |
| Peptides-func | LRG | LRGB 官方 |
| Peptides-struct | LRG | LRGB 官方 |