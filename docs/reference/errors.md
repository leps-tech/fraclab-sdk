# 错误处理

## 常见错误来源

- Bundle 结构不完整
- `ds.json` / `drs.json` 哈希不一致
- 输入参数不符合 InputSpec
- 输出结果不符合 OutputContract
- 算法运行时抛异常

## 常见异常类型

| 异常 | 说明 |
|------|------|
| `SnapshotError` | Snapshot 导入或访问失败 |
| `AlgorithmError` | 算法包导入或访问失败 |
| `RunError` | 运行创建或执行失败 |
| `OutputContainmentError` | 输出文件路径逃逸 `ctx.output.dir` |

## 常见 CLI 错误

- `manifest.json not found`
- `ds.json hash mismatch`
- `drs.json not found`
- `run manifest validation failed`

## 调试建议

1. 先跑 `fraclab-sdk validate bundle`
2. 再跑 `fraclab-sdk validate inputspec`
3. 再跑 `fraclab-sdk validate output-contract`
4. 运行后看：
   - `fraclab-sdk run logs <run_id>`
   - `fraclab-sdk result list <run_id>`

需要完整堆栈时，使用：

```bash
fraclab-sdk --debug <subcommand>
```
