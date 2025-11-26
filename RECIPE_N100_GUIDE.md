# RECIPE N=100 快速实验指南

## 🚀 启动实验

```bash
cd /root/REPAIR
./run_recipe_n100.sh
```

## ⏱️ 预计时间

- **RECIPE训练**: 1-2小时
- **RECIPE测试**: 10-15分钟
- **总计**: ~1.5-2.5小时

## 📊 实验内容

1. 安装RECIPE依赖
2. 训练RECIPE模型（ZsRE数据集）
3. 测试RECIPE @ N=100
4. 保存结果

## 📁 输出位置

```
recipe_experiment_N100/YYYYMMDD_HHMMSS/
├── logs/
│   ├── main.log      # 主日志
│   ├── train.log     # 训练详细日志
│   └── test.log      # 测试日志
└── results/          # RECIPE结果
```

## 🔍 监控命令

```bash
# 主日志
tail -f recipe_experiment_N100/*/logs/main.log

# 训练进度
tail -f recipe_experiment_N100/*/logs/train.log

# GPU使用
watch -n 1 nvidia-smi
```

## ✅ 完成后

结果将保存在 `recipe_experiment_N100/*/results/`

可以与之前的REPAIR @ N=100结果对比：
- REPAIR结果: `rebuttal_experiments/20251124_170554/`
- RECIPE结果: `recipe_experiment_N100/*/results/`

## 💡 注意事项

1. **训练时间**: 首次训练需要1-2小时
2. **Checkpoint**: 自动保存在 `/root/RECIPE_baseline/train_records/`
3. **可中断**: RECIPE支持从checkpoint恢复
