# InputSpec

`schema/inputspec.py` 用来声明算法接受哪些参数，以及 Workbench 该怎么渲染这些参数。

## 基本规则

- 必须导出 `INPUT_SPEC`
- 推荐继承 `schema/base.py` 里的 `CamelModel`
- 算法运行时的 `ctx.params` 是 `dict[str, Any]`
- `params.schema.json` 使用 camelCase

## 最小示例

```python
from .base import CamelModel, Field


class InputParams(CamelModel):
    maxItems: int = Field(default=10)
    datasetKey: str = Field(default="wells")


INPUT_SPEC = InputParams
```

对应运行时代码：

```python
def run(ctx):
    max_items = ctx.params.get("maxItems", 10)
    dataset_key = ctx.params.get("datasetKey", "wells")
```

## Workbench 数值精度

`number` 字段在 Workbench 的显示精度按 `step` 推导：

- `step=0.01` -> 2 位小数
- `step=0.001` -> 3 位小数
- 没有 `step` -> 按整数样式显示

```python
threshold: float = Field(default=0.5, json_schema_extra={"step": 0.01})
```

## Workbench 数组渲染边界

当前 Workbench Run 页不会为所有 `array` 字段自动生成结构化控件。

- `uiType="time_window"`: 有专门时间窗控件
- `array` of `enum`: 会渲染成多选
- 其他数组: 会退回 raw JSON 编辑

这类 schema 本身仍然是合法的，但校验器会给出 `INPUTSPEC_WORKBENCH_ARRAY_JSON_ONLY` 警告，提醒你 Workbench 端不是结构化表单体验。

## 时间窗参数

时间窗字段统一用：

`Optional[list[TimeWindow]]`

推荐直接使用 `time_window_list()` helper。

```python
from .base import CamelModel, Field, schemaExtra, time_window_list


WindowsTemplate = time_window_list(
    min_items=1,
    max_items=3,
    title="Windows",
    description="Template windows.",
)


class INPUT_SPEC(CamelModel):
    timeWindows_fracRecord_stage_5712: WindowsTemplate = Field(
        default=None,
        title="Time Windows",
        json_schema_extra=schemaExtra(
            uiType="time_window",
            unit="us",
            bindDatasetKey="fracRecord_stage_5712",
        ),
    )
```

关键规则：

- `bindDatasetKey` 必须和 `dist/drs.json` 里的 `datasets[*].key` 对齐
- `time_window_list()` 自带 `Optional`，不要再额外包一层
- `unit` 必须显式写成 `us`；运行时时间窗固定为 `us` 数值，不做秒/毫秒/日期时间转换
- 统一选择器会按当前 dataset 切换约束

## JSON 命名边界

要分清两类 JSON：

- 算法自己定义的：`params.schema.json`、`output_contract.json`、运行输出 `manifest.json`
  - 用 camelCase
- Bundle 自带的：`ds.json`、`drs.json`
  - 跟随平台原始格式，使用 `key` / `resource`

这两条不要混在一起。
