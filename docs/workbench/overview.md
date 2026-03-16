# Workbench 概览

## 启动

```bash
fraclab-workbench
```

或：

```bash
python -m fraclab_sdk.workbench
```

## 语言

Workbench 支持英文和简体中文。

- 页面右上角可以切换语言
- 当前语言会持久化
- 也可以通过 `FRACLAB_WORKBENCH_LANG=en|zh-CN` 设置初始语言

## 页面

| 页面 | 用途 |
|------|------|
| Home | 首页和资源概览 |
| Snapshots | 导入和管理快照 |
| Browse | 浏览 Bundle / Snapshot 数据 |
| Selection | 配置运行输入选择 |
| Run | 编辑参数并执行运行 |
| Results | 查看运行结果和日志 |
| Algorithm Edit | 编辑 `main.py` |
| Schema Edit | 编辑 `schema/inputspec.py` |
| Output Edit | 编辑 `schema/output_contract.py` |
| Export Algorithm | 编译并导出算法 |

## Browse 页面

Browse 页面现在按 dataset 和 item 浏览：

- `object_ndjson_lines` 以表格展示
- `frame_parquet_item_dirs` 支持文件预览和只读可视化

Parquet 预览会识别常见时间列并做必要的降采样。

## Run 页面

Run 页面会：

- 渲染 InputSpec 对应的参数控件
- 对多个时间窗字段使用统一时间窗选择器
- 在执行前生成当前 run 的输入目录

算法输出不会由页面手写 manifest，而是由 runner 结束时统一生成。
