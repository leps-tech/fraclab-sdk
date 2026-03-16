# CLI 概览

## 运行闭环黄金路径

```bash
# 1. 导入快照和算法
fraclab-sdk snapshot import /path/to/bundle
fraclab-sdk algo import ./my-algorithm.zip

# 2. 查看资源
fraclab-sdk snapshot list
fraclab-sdk algo list

# 3. 创建并执行 run
fraclab-sdk run create <snapshot_id> <algorithm_name> --params params.json
fraclab-sdk run execute <run_id>

# 4. 查看结果
fraclab-sdk result list <run_id>
fraclab-sdk run logs <run_id>
```

## 命令分组

### 算法

- `fraclab-sdk algo init`
- `fraclab-sdk algo compile`
- `fraclab-sdk algo export`
- `fraclab-sdk algo import`
- `fraclab-sdk algo list`

### 快照

- `fraclab-sdk snapshot import`
- `fraclab-sdk snapshot list`

### 运行

- `fraclab-sdk run create`
- `fraclab-sdk run execute`
- `fraclab-sdk run list`
- `fraclab-sdk run logs`

### 结果

- `fraclab-sdk result list`

### 校验

- `fraclab-sdk validate bundle`
- `fraclab-sdk validate inputspec`
- `fraclab-sdk validate output-contract`
- `fraclab-sdk validate run-manifest`

## 调试

需要完整堆栈时：

```bash
fraclab-sdk --debug <subcommand>
```
