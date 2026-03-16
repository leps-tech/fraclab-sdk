# 编译与导入导出

## 算法工作区结构

最小工作区：

```text
my-algorithm/
  main.py
  manifest.json
  schema/
    base.py
    inputspec.py
    output_contract.py
```

编译后会生成：

```text
dist/
  params.schema.json
  output_contract.json
  ds.json
  drs.json
```

## 编译

```bash
fraclab-sdk algo compile ./my-algorithm --bundle /path/to/bundle
```

编译会做这些事：

- 从 `schema/inputspec.py` 生成 `dist/params.schema.json`
- 从 `schema/output_contract.py` 生成 `dist/output_contract.json`
- 从 Bundle 复制 `ds.json` 和 `drs.json`

## 导出

最推荐：

```bash
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle
```

说明：

- `export` 需要完整 `dist/`
- `--auto-compile` 会在缺失时先编译
- 如果编译需要 DS/DRS，就必须提供 `--bundle`

## 导入算法包

```bash
fraclab-sdk algo import ./my-algorithm.zip
```

导入时会检查：

- `manifest.json` 是否存在
- 包内路径是否安全
- 关键文件是否齐全

## 导入 Snapshot

```bash
fraclab-sdk snapshot import /path/to/bundle
```

导入时会检查：

- Bundle 路径结构
- `manifest.json` 路径声明
- `ds.json` / `drs.json` 哈希
- `data/` 是否存在

## 常见失败原因

- `manifest.json not found`
- `ds.json hash mismatch`
- `drs.json not found`
- 导出时缺少 `dist/params.schema.json` 或 `dist/output_contract.json`
- `bindDatasetKey` 和 DRS 的 `datasets[*].key` 不一致
