# Fraclab SDK Reference

> 版本: 0.1.3
> Python: >=3.11

Fraclab SDK 是一个算法开发与执行框架，帮助算法开发者快速构建、测试和部署数据处理算法。

---

## 目录

- [1. 安装](#installation)
- [2. 快速开始：编写你的第一个算法（从 Bundle 到运行）](#quickstart)
  - [2.1 开发闭环（先看这个）](#quickstart-flow)
  - [2.2 准备 Bundle 路径](#quickstart-bundle-path)
  - [2.3 编写算法入口 `main.py`](#quickstart-main)
  - [2.4 定义 `InputSpec`](#quickstart-inputspec)
  - [2.5 定义 `OutputContract`](#quickstart-output-contract)
  - [2.6 创建算法清单](#quickstart-manifest)
  - [2.7 用 Bundle 编译并导出算法包](#quickstart-build-export)
  - [2.8 导入并运行算法包](#quickstart-run)
  - [2.9 `SelectionModel` 与 `run_ds` 的关系](#quickstart-selection-runds)
- [3. Bundle 与 Snapshot（概念与关系）](#bundle-and-snapshot)
- [4. 算法开发详解](#algorithm-development-guide)
- [5. CLI 命令行工具](#cli-tools)
- [6. SDK 内部模块](#sdk-internal-modules)
- [7. 数据模型](#data-models)
- [8. 错误处理](#error-handling)
- [9. 安全特性](#security-features)
- [10. 完整示例](#complete-examples)
- [11. 附录 A: Bundle 结构详解](#appendix-a-bundle-structure)

---

<a id="installation"></a>
## 1. 安装

轻量安装（核心 SDK / CLI，自动带上科学计算依赖）：

```bash
pip install fraclab-sdk
```

安装并启用 Workbench UI：

```bash
pip install "fraclab-sdk[workbench]"
fraclab-workbench           # CLI entry point
# 或
python -m fraclab_sdk.workbench
```

### 依赖说明

- 核心安装会自动安装并锁定：`numpy>=2.1.1`, `pandas>=2.2.3`, `scipy>=1.13.1`, `matplotlib>=3.9.2`, `fastapi>=0.115.0`, `rich>=13.9.0`。无需手动再装，避免版本冲突。
- 可选 `workbench` 额外安装 UI 依赖：`streamlit>=1.30`, `pyarrow>=16.0.0`（`pandas` 已在核心里）。

---

<a id="quickstart"></a>
## 2. 快速开始：编写你的第一个算法（从 Bundle 到运行）

<a id="quickstart-flow"></a>
### 2.1 开发闭环（先看这个）

对初次使用者，先记住这一条主线：

1. 你拿到平台给的 **Bundle** 目录（含 `manifest.json`、`ds.json`、`drs.json`、`data/`）
2. 你编写算法源码（`main.py` + `schema/*` + `manifest.json`）
3. 你用 Bundle 编译，生成 `dist/*.json`（尤其 `dist/ds.json` 与 `dist/drs.json`）
4. 你导出算法 zip
5. 你导入 Snapshot（来自 Bundle）和算法 zip
6. 你创建 run、执行 run、查看结果

推荐一条命令完成“编译 + 导出”：

```bash
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle
```

说明（与代码行为一致）：
- `export` 本身要求 `dist/params.schema.json`、`dist/output_contract.json`、`dist/ds.json`、`dist/drs.json` 已存在
- `--auto-compile` 会在缺少这些文件时自动调用 compile
- `compile` 阶段如果没有可用的 `dist/ds.json` / `dist/drs.json`，就必须提供 `--bundle`（用于复制 bundle 的 `ds.json` 与 `drs.json`）

<a id="quickstart-bundle-path"></a>
### 2.2 准备 Bundle 路径

Bundle 是算法开发和运行的共同输入。你至少会在两个阶段用到它：

- **编译阶段**：提供 DS/DRS（生成 `dist/ds.json` 与 `dist/drs.json`）
- **运行阶段**：导入为 Snapshot，作为 run 的数据来源

最小检查：

```bash
fraclab-sdk validate bundle /path/to/bundle
```

<a id="quickstart-main"></a>
### 2.3 编写算法入口 `main.py`

算法开发者主要使用 `fraclab_sdk.runtime` 模块中的两个核心类：

```python
from fraclab_sdk.runtime import DataClient, ArtifactWriter
```

- **DataClient**: 读取输入数据
- **ArtifactWriter**: 写入输出结果

#### 入口签名与模板

创建 `main.py` 作为算法入口文件。

#### 入口函数签名约定

**算法入口函数必须严格遵循以下签名:**

```python
def run(ctx):
    ...
```

| 约定 | 要求 | 说明 |
|------|------|------|
| 文件名 | `main.py` | 必须是 `main.py`，不能是其他名称 |
| 函数名 | `run` | 必须是 `run`，区分大小写 |
| 参数 | 仅 `ctx` 一个参数 | SDK 使用 `module.run(ctx)` 调用，**不支持**其他签名 |
| 返回值 | 无要求 | 返回值会被忽略 |

> **警告**: 以下写法都会导致运行失败:
> ```python
> # ❌ 错误: 参数名不影响，但参数个数必须是 1
> def run(ctx, extra_arg):  # TypeError: run() missing required argument
>
> # ❌ 错误: 函数名错误
> def execute(ctx):  # AttributeError: module has no attribute 'run'
>
> # ❌ 错误: 放在其他文件
> # helper.py 中定义 run()  # 不会被加载
> ```

#### 最小可运行模板

```python
# main.py
def run(ctx):
    """算法入口函数 - 最小模板。

    Args:
        ctx: RunContext，包含:
            - ctx.data_client: DataClient 实例
            - ctx.params: dict[str, Any]，用户参数
            - ctx.artifacts: ArtifactWriter 实例
            - ctx.logger: logging.Logger 实例
            - ctx.run_context: dict，运行上下文
    """
    logger = ctx.logger
    logger.info("算法开始执行")

    # 你的逻辑...

    logger.info("算法执行完成")
```

#### 完整示例

```python
# main.py
def run(ctx):
    """算法入口函数。

    Args:
        ctx: RunContext，包含:
            - ctx.data_client: DataClient 实例
            - ctx.params: dict，用户参数
            - ctx.artifacts: ArtifactWriter 实例
            - ctx.logger: Logger 实例
            - ctx.run_context: dict，运行上下文
    """
    dc = ctx.data_client
    aw = ctx.artifacts
    params = ctx.params
    logger = ctx.logger

    # 获取参数
    threshold = params.get("threshold", 0.5)
    logger.info(f"开始处理，阈值: {threshold}")

    # 读取输入数据
    for dataset_key in dc.get_dataset_keys():
        count = dc.get_item_count(dataset_key)
        logger.info(f"数据集 {dataset_key} 包含 {count} 个项目")

        for i in range(count):
            # 读取 NDJSON 对象
            obj = dc.read_object(dataset_key, i)

            # 处理数据...
            result = process(obj, threshold)

            # 写入结果
            aw.write_scalar(f"{dataset_key}_result_{i}", result)

    # 写入汇总结果
    aw.write_json("summary", {"status": "completed", "threshold": threshold})

    logger.info("算法执行完成")

def process(data, threshold):
    """你的数据处理逻辑"""
    return data.get("value", 0) > threshold
```

<a id="quickstart-inputspec"></a>
### 2.4 定义输入参数规格 (InputSpec)

创建 `schema/inputspec.py` 定义算法接受的参数：

```python
# schema/inputspec.py
from pydantic import BaseModel, Field

class InputParams(BaseModel):
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="检测阈值"
    )

    debug: bool = Field(
        default=False,
        description="启用调试模式"
    )

# 必须导出 INPUT_SPEC
INPUT_SPEC = InputParams
```

#### ctx.params 的类型与访问方式

**`ctx.params` 是 `dict[str, Any]` 类型**，键名来自 `params.json` (JSON 原始键名)。

```python
# main.py - 算法代码
def run(ctx):
    # ctx.params 是 dict，使用 dict 访问方式
    threshold = ctx.params.get("threshold", 0.5)
    debug = ctx.params.get("debug", False)

    # 嵌套对象同样是 dict
    filters = ctx.params.get("filters", {})
    min_depth = filters.get("minDepth", 0)
```

#### InputSpec 与 JSON 命名规则

| 层级 | 命名风格 | 示例 | 说明 |
|------|---------|------|------|
| **InputSpec 定义** | `snake_case` | `max_items` | Pydantic 字段名 |
| **JSON / params.json** | `camelCase` | `maxItems` | 使用 `alias_generator=to_camel` 时 |
| **算法访问 ctx.params** | JSON 原始键名 | `ctx.params["maxItems"]` | dict 访问，键名与 JSON 一致 |

**InputSpec 定义示例：**

```python
# schema/inputspec.py
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

class MyParams(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,  # snake_case -> camelCase
    )

    max_items: int = Field(default=10)      # Python: max_items
    dataset_key: str = Field(default="wells")  # Python: dataset_key

INPUT_SPEC = MyParams
```

**对应的 params.json：**

```json
{
  "maxItems": 5,
  "datasetKey": "stages"
}
```

**算法中访问：**

```python
def run(ctx):
    # ctx.params 是 dict，键名与 JSON 一致 (camelCase)
    max_items = ctx.params.get("maxItems", 10)
    dataset_key = ctx.params.get("datasetKey", "wells")
```

> **提示**: 如果 InputSpec 没有配置 `alias_generator`，则 JSON 和 ctx.params 键名都使用 `snake_case`。

#### 数值参数精度（Workbench）

在 Workbench 的参数输入 UI 中，`number` 类型字段的显示精度按 `InputSpec` 生成的 schema `step` 决定：

- 有 `step`：小数位数由 `step` 推导（例如 `step=0.01` 显示 2 位，`step=0.001` 显示 3 位）
- 无 `step`：按整数显示（不显示小数位）

示例：

```python
from pydantic import BaseModel, Field

class InputParams(BaseModel):
    threshold: float = Field(default=0.5, json_schema_extra={"step": 0.01})  # 2 位小数
    gain: float = Field(default=1.0)  # 未设置 step，Workbench 按整数样式显示
```

#### 时间窗参数（`ui_type="time_window"`，新版）

Run 页面使用一个统一时间窗选择器（位于参数区底部），在组件内切换 dataset。

关键规则：
- 字段 shape：`List[TimeWindow]` 或 `Optional[List[TimeWindow]]`（`TimeWindow = {min,max}`）。
- 每个时间窗字段必须配置 `bind_dataset_key`（顶层或 `json_schema_extra`）。
- 匹配依据是当前 run 的 `input/ds.json` 中已选 dataset keys，不是 params 里的 `datasetKey` 字段。
- 同一个选择器在切换 dataset 时会套用该 dataset 对应字段的窗口约束（`minItems/maxItems`）。
- 若定义 `window_slots` 与 `window_slot_fallback_note`，图像上方会显示“下一时间窗”备注：
  - 还未选窗：显示第 1 条备注
  - 选完第 1 个窗后：显示第 2 条备注
  - 依次类推；超出后显示 fallback 备注

<a id="quickstart-output-contract"></a>
### 2.5 定义输出合约 (OutputContract)

创建 `schema/output_contract.py` 声明算法的输出结构：

```python
# schema/output_contract.py

OUTPUT_CONTRACT = {
    "datasets": [
        {
            "key": "metrics",
            "kind": "scalar",
            "owner": "well",
            "cardinality": "many",
            "required": True,
            "dimensions": ["stage"],
            "schema": {"type": "scalar", "dtype": "float"},
            "role": "primary",
            "description": "每个井/阶段的评估指标"
        },
        {
            "key": "summary",
            "kind": "object",
            "owner": "platform",
            "cardinality": "one",
            "required": True,
            "dimensions": [],
            "schema": {"type": "object"},
            "role": "primary",
            "description": "汇总结果"
        },
        {
            "key": "debug_plots",
            "kind": "blob",
            "owner": "well",
            "cardinality": "many",
            "required": False,
            "dimensions": [],
            "schema": {"type": "blob", "mime": "image/png"},
            "role": "debug",
            "description": "调试图表 (可选)"
        }
    ]
}
```

#### OutputContract 字段规范

| 字段 | 必填 | 类型 | 可选值 | 说明 |
|------|------|------|--------|------|
| `key` | **是** | string | - | 数据集唯一键名 |
| `kind` | **是** | string | `"scalar"` / `"object"` / `"blob"` / `"frame"` | 数据类型 |
| `owner` | **是** | string | `"stage"` / `"well"` / `"platform"` | 所有者级别 |
| `cardinality` | 否 | string | `"one"` / `"many"` | 项目数量约束，默认 `"many"` |
| `required` | 否 | bool | - | 是否必须产出，默认 `true` |
| `dimensions` | 否 | string[] | - | 维度键列表 |
| `schema` | **是** | object | - | 数据 schema |
| `schema.type` | **是** | string | 与 `kind` 对应 | schema 类型标识 |
| `role` | 否 | string | `"primary"` / `"supporting"` / `"debug"` | 输出角色 |
| `description` | 否 | string | - | 描述说明 |

#### kind 与 schema.type 对应关系

| kind | schema.type | ArtifactWriter 方法 | 说明 |
|------|-------------|---------------------|------|
| `"scalar"` | `"scalar"` | `write_scalar()` | 标量值 (数字/字符串/布尔) |
| `"object"` | `"object"` | `write_json()` | JSON 对象 |
| `"blob"` | `"blob"` | `write_blob()` / `write_file()` | 二进制文件 |
| `"frame"` | `"frame"` | (暂不支持) | 表格数据 |

#### owner 级别说明

| owner | 含义 | ArtifactWriter owner 参数 |
|-------|------|--------------------------|
| `"platform"` | 平台级 (全局) | `owner={"platformId": "..."}` |
| `"well"` | 井级 | `owner={"wellId": "..."}` |
| `"stage"` | 阶段级 | `owner={"stageId": "..."}` |

#### cardinality 约束

| cardinality | 含义 | 验证规则 |
|-------------|------|----------|
| `"one"` | 恰好一个项目 | required=true 时必须 1 项; required=false 时最多 1 项 |
| `"many"` | 一个或多个 | required=true 时至少 1 项; required=false 时 0 项或多项 |

#### dimensions 使用

当数据集有维度约束时，写入时必须提供对应的 `dims`:

```python
# OutputContract 定义: dimensions: ["stage", "iteration"]
aw.write_scalar(
    "loss",
    0.05,
    dataset_key="training_metrics",
    owner={"wellId": "W001"},
    dims={"stage": 1, "iteration": 100}  # 必须包含所有定义的维度
)
```

<a id="quickstart-manifest"></a>
### 2.6 创建算法清单

创建 `manifest.json` — 这是**打包、导入、发布的唯一标准清单**：

```json
{
  "manifestVersion": "1",
  "algorithmId": "my-algorithm",
  "name": "My Algorithm",
  "summary": "算法简短描述 (必填)",
  "notes": "详细说明、使用注意事项等 (可选)",
  "tags": ["analysis", "well-log"],
  "authors": [
    {
      "name": "张三",
      "email": "zhangsan@example.com",
      "organization": "示例公司"
    }
  ],
  "contractVersion": "1.0.0",
  "codeVersion": "1.0.0",
  "files": {
    "paramsSchemaPath": "dist/params.schema.json",
    "outputContractPath": "dist/output_contract.json",
    "dsPath": "dist/ds.json",
    "drsPath": "dist/drs.json"
  },
  "requires": {
    "sdk": "0.1.3",
    "core": "1.0.0"
  },
  "repository": "https://github.com/example/my-algorithm",
  "homepage": "https://example.com/my-algorithm",
  "license": "MIT"
}
```

#### 字段规范详解

| 字段 | 必填 | 类型 | 约束 | 说明 |
|------|------|------|------|------|
| `manifestVersion` | **是** | `"1"` | 固定值 | 清单版本 |
| `algorithmId` | **是** | string | 1-128 字符 | 算法唯一标识符 (用于导入/引用) |
| `name` | **是** | string | 1-256 字符 | 算法显示名称 |
| `summary` | **是** | string | 1-256 字符 | 简短描述 |
| `notes` | 否 | string | - | 详细说明 |
| `tags` | 否 | string[] | 每项 1-256 字符 | 标签列表 |
| `authors` | **是** | Author[] | 至少 1 项 | 作者列表 |
| `authors[].name` | **是** | string | 1-256 字符 | 作者姓名 |
| `authors[].email` | 否 | string | 3-320 字符 | 邮箱地址 |
| `authors[].organization` | 否 | string | 1-256 字符 | 所属组织 |
| `contractVersion` | **是** | string | SemVer 格式 | 输出合约版本 (如 `1.0.0`) |
| `codeVersion` | **是** | string | - | 代码版本 (用作算法版本标识) |
| `files` | **是** | object | - | 产物文件路径 (见下表) |
| `requires` | 否 | object | - | 兼容性要求 |
| `requires.sdk` | 否 | string | SemVer 格式 | SDK 最低版本 |
| `requires.core` | 否 | string | SemVer 格式 | Core 最低版本 |
| `repository` | 否 | string | 1-2048 字符 | 代码仓库 URL |
| `homepage` | 否 | string | 1-2048 字符 | 主页 URL |
| `license` | 否 | string | 1-256 字符 | 许可证标识 |

#### files 字段详解

`files` 用于指定编译产物的位置，导入时 SDK 根据此字段定位文件：

| 字段 | 说明 |
|------|------|
| `files.paramsSchemaPath` | 参数 JSON Schema 路径（导入阶段必需） |
| `files.dsPath` | DS 文件路径（导出阶段会自动补齐） |
| `files.drsPath` | DRS 文件路径（导出阶段会自动补齐） |
| `files.outputContractPath` | 输出合约路径（建议提供；缺失时运行阶段可跳过合约校验） |

**路径规则:**
- 所有路径均为**相对于算法包根目录**的路径
- SDK 导入最低要求是 `files.paramsSchemaPath`
- `files.dsPath` / `files.drsPath` / `files.outputContractPath` 可选（建议提供）
- 推荐使用 `dist/` 前缀 (如 `dist/params.schema.json`)

#### 导入阶段：SDK 最小算法包要求

导入算法包 (zip 或目录) 时，SDK 验证以下文件:

| 文件 | 必须 | 说明 |
|------|------|------|
| `main.py` | **是** | 算法入口文件，必须包含 `run(ctx)` 函数 |
| `manifest.json` | **是** | 算法清单 (含 `files.*Path` 字段) |
| `dist/params.schema.json` | **是** | 参数 JSON Schema (路径由 `files.paramsSchemaPath` 指定) |
| `dist/ds.json` | 否 | 数据规格（通常在导出阶段从 Bundle 注入） |
| `dist/drs.json` | 否 | 数据需求规格（通常在导出阶段从 Bundle 注入） |
| `dist/output_contract.json` | 否 | 输出合约（建议提供） |

> **重要**: 文件实际位置由 `manifest.json` 的 `files.*Path` 字段决定。

#### 导入阶段常见失败原因

1. **`manifest.json not found`**: 包内缺少 manifest.json，或 zip 解压后目录结构嵌套
2. **`main.py not found`**: 入口文件缺失
3. **`dist/params.schema.json not found`**: 未执行 `fraclab-sdk algo compile`
5. **`contractVersion must be semver-like`**: contractVersion 格式错误，应为 `x.y.z`
6. **`authors must contain at least one author`**: authors 列表为空

#### 导出阶段要求（发布包）

`fraclab-sdk algo export ...` 会要求完整 `dist` 产物：

- `dist/params.schema.json`
- `dist/output_contract.json`
- `dist/ds.json`
- `dist/drs.json`

推荐使用：

```bash
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle
```

导出页会从所选 Bundle 注入 `ds/drs`，并自动补齐 `files.dsPath` / `files.drsPath`（若缺失）。

### 2.6.1 项目结构

完整的算法工作区结构：

```
my-algorithm/
├── manifest.json           # 算法清单 (导出包必须)
├── main.py                 # 算法入口 (必须包含 run 函数)
├── schema/
│   ├── __init__.py
│   ├── inputspec.py        # INPUT_SPEC 定义
│   └── output_contract.py  # OUTPUT_CONTRACT 定义
├── lib/                    # 可选: 算法依赖模块
│   └── utils.py
└── dist/                   # 编译产物 (自动生成)
    ├── params.schema.json  # 从 INPUT_SPEC 编译
    ├── output_contract.json # 从 OUTPUT_CONTRACT 编译
    ├── ds.json             # 从 Bundle 复制
    └── drs.json            # 从 Bundle 复制
```

**导出后的算法包结构** (zip 内或目录):

```
my-algorithm.zip/
├── manifest.json           # 必须: 算法清单 (含 files.*Path)
├── main.py                 # 必须: 入口文件
├── dist/                   # 编译产物目录
│   ├── params.schema.json  # 必须: 参数 Schema
│   ├── ds.json             # 必须: 数据规格
│   ├── drs.json            # 必须: 数据需求规格
│   └── output_contract.json # 必须: 输出合约
├── schema/                 # 可选: schema 源码
│   ├── __init__.py
│   ├── inputspec.py
│   └── output_contract.py
└── README.md               # 可选: 说明文件
```

<a id="quickstart-build-export"></a>
### 2.7 用 Bundle 编译并导出算法包

推荐命令（单命令闭环）：

```bash
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle
```

为什么推荐这一条：

1. 最终可导入运行的产物必须是包含 `dist/*.json` 的算法包
2. `export` 只负责打包，不会凭空生成缺失的 `dist/*.json`
3. `--auto-compile --bundle` 能在需要时先补齐 `dist/*.json`，再打包

等价的两步写法（功能上成立，但更容易漏步骤）：

```bash
fraclab-sdk algo compile ./my-algorithm --bundle /path/to/bundle
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip
```

注意：
- 如果你已经有有效的 `dist/ds.json` 与 `dist/drs.json`，`compile` 可以不传 `--bundle`
- 但对首次构建，通常都应显式传 `--bundle`

<a id="quickstart-run"></a>
### 2.8 导入并运行算法包

```bash
# 1) 导入 Bundle -> Snapshot
fraclab-sdk snapshot import /path/to/bundle

# 2) 导入算法 zip
fraclab-sdk algo import ./my-algorithm.zip

# 3) 创建并执行 run
fraclab-sdk run create <snapshot_id> <algorithm_id> <version> --params params.json
fraclab-sdk run exec <run_id> --timeout 300

# 4) 查看结果
fraclab-sdk results list <run_id>
```

<a id="quickstart-selection-runds"></a>
### 2.9 `SelectionModel` 与 `run_ds` 的关系

这是 run 侧最容易混淆的点：

- `SelectionModel`：你给 SDK 的“选择意图”（基于 snapshot 索引选哪些 item）
- `run_ds`：SDK 根据 selection 生成的“运行时 DataSpec 子集”（会重建为 0..N-1，并保留 `sourceItemIndex` 映射）

关系链（代码路径）：

1. `selection = SelectionModel.from_snapshot_and_drs(snapshot, algorithm.drs)`
2. 你调用 `run_mgr.create_run(..., selection=selection, ...)`
3. `create_run()` 内部调用 `selection.build_run_ds()`
4. `Materializer` 用 `run_ds` 物化 `runs/<run_id>/input/`

结论：
- **SDK 调用方通常只需要管理 `SelectionModel`，不用手动传 `run_ds`**
- `run_ds` 是运行前的内部物化输入模型；你可在调试/诊断时显式查看它

---

<a id="bundle-and-snapshot"></a>
## 3. Bundle 与 Snapshot（概念与关系）

### 3.1 Bundle 是什么

Bundle 是平台提供的原始数据包目录，至少包含：

- `manifest.json`
- `ds.json`
- `drs.json`
- `data/`

Bundle 的两个用途：

1. 给算法编译提供 DS/DRS（拷贝到 `dist/ds.json`、`dist/drs.json`）
2. 导入 SDK 后生成 Snapshot，供 run 选择与执行

### 3.2 Snapshot 是什么

Snapshot 是 Bundle 导入 SDK 后形成的内部快照副本（默认在 `~/.fraclab/snapshots/<snapshot_id>`）。

- 导入命令：`fraclab-sdk snapshot import /path/to/bundle`
- 运行时你用的是 `snapshot_id`，不是原始 bundle 路径
- `run create` 阶段会基于 snapshot + selection 生成 run 输入

### 3.3 Bundle / Snapshot / RunInput 的关系

```text
Bundle (原始数据包)
  -> snapshot import
Snapshot (SDK库内快照)
  + SelectionModel (选中的 snapshot item 索引)
  -> build_run_ds()
Run Input (runs/<run_id>/input: ds.json/drs.json/data/)
```

其中 `run_ds` 是 “Run Input 里的 ds.json 对应对象”，不是 Bundle 的 `ds.json` 原样复制。

### 3.4 为什么不能随便改 Bundle

Snapshot 导入与校验依赖 `manifest.json` 中的哈希字段（`dsSha256`、`drsSha256`）。
任何对 `ds.json`/`drs.json` 的手工修改都会触发校验失败。

```bash
fraclab-sdk validate bundle /path/to/bundle
```

常见错误：
- `ds.json hash mismatch`: 文件被改动或损坏
- `drs.json not found`: Bundle 不完整
- `manifest.json not found`: 非有效 Bundle 目录

> 详细目录与字段见 [11. 附录 A: Bundle 结构详解](#appendix-a-bundle-structure)

---

<a id="algorithm-development-guide"></a>
## 4. 算法开发详解

### DataClient - 读取输入数据

`DataClient` 提供统一的数据读取接口。

```python
from fraclab_sdk.runtime import DataClient
from pathlib import Path

dc = DataClient(Path("input"))
```

#### 获取数据集信息

```python
# 获取所有数据集键
keys = dc.get_dataset_keys()  # ["wells", "frames", ...]

# 获取数据集中的项目数量
count = dc.get_item_count("wells")  # 10

# 获取数据集布局类型
layout = dc.get_layout("wells")  # "object_ndjson_lines" 或 "frame_parquet_item_dirs"
```

#### 读取 NDJSON 数据

用于 `layout="object_ndjson_lines"` 的数据集：

```python
# 读取单个对象 (按索引)
obj = dc.read_object("wells", 0)  # 返回 dict

# 迭代所有对象
for idx, obj in dc.iterate_objects("wells"):
    print(f"Item {idx}: {obj}")
```

#### 读取 Parquet 数据

用于 `layout="frame_parquet_item_dirs"` 的数据集：

```python
# 获取 parquet 文件目录
parquet_dir = dc.get_parquet_dir("frames", 0)

# 获取所有 parquet 文件列表
parquet_files = dc.get_parquet_files("frames", 0)

# 使用 pandas/polars 读取
import pandas as pd
df = pd.read_parquet(parquet_dir)
```

### ArtifactWriter - 写入输出结果

`ArtifactWriter` 提供安全的输出写入机制，自动防止路径逃逸攻击。

```python
from fraclab_sdk.runtime import ArtifactWriter
from pathlib import Path

aw = ArtifactWriter(Path("output"))
```

#### 写入标量值

```python
# 基本用法
aw.write_scalar("score", 0.95)
aw.write_scalar("count", 42)
aw.write_scalar("name", "result_a")

# 指定数据集和所有者
aw.write_scalar(
    "accuracy",
    0.87,
    dataset_key="metrics",
    owner={"wellId": "W001"},
    dims={"stage": 1},
    meta={"unit": "percent"}
)
```

#### 写入 JSON

```python
# 基本用法
path = aw.write_json("metrics", {"accuracy": 0.95, "loss": 0.05})

# 自定义文件名
path = aw.write_json("results", data, filename="analysis_results.json")

# 完整参数
path = aw.write_json(
    "summary",
    {"status": "ok"},
    filename="summary.json",
    dataset_key="outputs",
    owner={"platformId": "P001"}
)
```

#### 写入二进制文件

```python
# 写入字节数据
image_bytes = generate_plot()
path = aw.write_blob(
    "plot",
    image_bytes,
    "plot.png",
    mime_type="image/png"
)

# 复制现有文件
path = aw.write_file(
    "report",
    Path("/tmp/generated_report.pdf"),
    filename="report.pdf",
    mime_type="application/pdf"
)
```

### ArtifactWriter 与 OutputContract/Manifest 映射关系

ArtifactWriter 的写入操作会自动生成 `output/manifest.json`，理解参数与输出的映射关系是正确使用的关键。

#### 参数映射表

| ArtifactWriter 参数 | manifest.json 字段 | OutputContract 字段 | 说明 |
|---------------------|-------------------|---------------------|------|
| `artifact_key` | `artifact.artifactKey` | - | 制品唯一标识 |
| `dataset_key` | `datasetKey` | `datasets[].key` | 数据集键，默认 `"artifacts"` |
| `owner` | `item.owner` | `datasets[].owner` | 所有者: `{wellId, stageId, platformId}` |
| `dims` | `item.dims` | `datasets[].dimensions` | 维度值字典 |
| `meta` | `item.meta` | - | 元数据 (manifest 专用) |
| `item_key` | `item.itemKey` | - | 项目键，默认等于 artifact_key |
| (写入类型) | `artifact.type` | `datasets[].kind` | 制品类型 |

#### 写入操作到 Manifest 的转换

```python
# 算法代码中的写入
aw.write_scalar(
    "accuracy",           # artifact_key
    0.95,                 # value
    dataset_key="metrics",
    owner={"wellId": "W001"},
    dims={"stage": 1},
    meta={"unit": "percent"}
)
```

生成的 `output/manifest.json` 片段:

```json
{
  "datasets": [
    {
      "datasetKey": "metrics",
      "items": [
        {
          "itemKey": "accuracy",
          "owner": { "wellId": "W001" },
          "dims": { "stage": 1 },
          "meta": { "unit": "percent" },
          "artifact": {
            "artifactKey": "accuracy",
            "type": "scalar",
            "value": 0.95
          }
        }
      ]
    }
  ]
}
```

#### 类型对应关系

| 写入方法 | manifest `artifact.type` | OutputContract `kind` | 说明 |
|----------|--------------------------|----------------------|------|
| `write_scalar()` | `"scalar"` | `"scalar"` | 标量值 (直接存 value) |
| `write_json()` | `"json"` | `"object"` | JSON 对象 (存 uri) |
| `write_blob()` | `"blob"` | `"blob"` | 二进制文件 (存 uri + mimeType) |
| `write_file()` | `"blob"` | `"blob"` | 复制文件 (存 uri + mimeType) |

#### OutputContract 定义与 ArtifactWriter 使用示例

**OutputContract 定义** (`schema/output_contract.py`):

```python
OUTPUT_CONTRACT = {
    "datasets": [
        {
            "key": "metrics",
            "kind": "scalar",
            "owner": "well",
            "cardinality": "many",
            "dimensions": ["stage"],
            "schema": {"type": "scalar", "dtype": "float"}
        },
        {
            "key": "reports",
            "kind": "blob",
            "owner": "well",
            "cardinality": "one",
            "schema": {"type": "blob", "mime": "application/pdf"}
        }
    ]
}
```

**对应的算法写入代码**:

```python
def run(ctx):
    aw = ctx.artifacts

    # 符合 "metrics" 数据集定义
    # owner="well" → 必须提供 wellId
    # dimensions=["stage"] → dims 必须包含 stage 键
    aw.write_scalar(
        "accuracy",
        0.95,
        dataset_key="metrics",
        owner={"wellId": "W001"},
        dims={"stage": 1}
    )

    # 符合 "reports" 数据集定义
    # cardinality="one" → 该数据集只能有一个项目
    aw.write_file(
        "report",
        Path("/tmp/report.pdf"),
        dataset_key="reports",
        owner={"wellId": "W001"},
        mime_type="application/pdf"
    )
```

#### 验证输出与合约一致性

```bash
# 验证运行输出是否符合合约
fraclab-sdk validate run-manifest output/manifest.json --contract dist/output_contract.json
```

验证检查项:
- 合约中所有 `required=true` 的数据集必须存在
- 数据集的 `cardinality` 约束 (one/many)
- `owner` 类型匹配 (well/stage/platform)
- `dimensions` 键集合匹配
- `kind` 与 `artifact.type` 兼容

### 日志记录

使用 `ctx.logger` 记录日志：

```python
def run(ctx):
    logger = ctx.logger

    logger.debug("调试信息")
    logger.info("常规信息")
    logger.warning("警告信息")
    logger.error("错误信息")
```

日志会同时输出到：
- 控制台 (INFO 及以上级别)
- `output/_logs/algorithm.log` 文件 (DEBUG 及以上级别)

---

<a id="cli-tools"></a>
## 5. CLI 命令行工具

安装后可使用 `fraclab-sdk` 命令。

### 运行闭环黄金路径

以下是从导入到执行的完整流程示例。

#### 1. 导入快照和算法

```bash
# 导入数据快照 (Bundle)
$ fraclab-sdk snapshot import /path/to/my-bundle
Imported snapshot: a1b2c3d4

# 导入算法包
$ fraclab-sdk algo import ./my-algorithm.zip
Imported algorithm: my-algorithm:1.0.0
```

#### 2. 查看已导入资源

```bash
# 列出快照
$ fraclab-sdk snapshot list
a1b2c3d4    my-bundle-v1    2024-01-15T10:30:00
e5f6g7h8    test-bundle     2024-01-14T09:00:00

# 列出算法
$ fraclab-sdk algo list
my-algorithm    1.0.0    2024-01-15T11:00:00
other-algo      2.1.0    2024-01-10T08:00:00
```

**ID 格式说明:**
- `snapshot_id`: 8 位十六进制字符串 (如 `a1b2c3d4`)
- `algorithm_id`: 算法的 algorithmId 字段值 (如 `my-algorithm`)
- `version`: 算法的 codeVersion 字段值，遵循 SemVer (如 `1.0.0`)

#### 3. 准备参数文件

创建 `params.json`:

```json
{
  "threshold": 0.8,
  "debug": false,
  "outputFormat": "detailed",
  "filters": {
    "minDepth": 1000,
    "maxDepth": 5000
  }
}
```

**参数文件要求:**
- JSON 格式，编码 UTF-8
- 键名对应 InputSpec 中定义的字段
- 未提供的字段使用 InputSpec 中的默认值

#### 4. 创建并执行运行

```bash
# 创建运行 (自动选择所有数据项)
$ fraclab-sdk run create a1b2c3d4 my-algorithm 1.0.0 --params params.json
f9e8d7c6

# 执行运行
$ fraclab-sdk run exec f9e8d7c6 --timeout 300
succeeded (exit_code=0)
```

#### 5. 查看结果

```bash
# 列出产出的制品
$ fraclab-sdk results list f9e8d7c6
Status: succeeded
accuracy    scalar
summary     json      file:///Users/.../output/artifacts/summary.json
report      blob      file:///Users/.../output/artifacts/report.pdf

# 查看运行日志
$ fraclab-sdk run tail f9e8d7c6
[INFO] 2024-01-15 12:00:00 - 开始处理，阈值: 0.8
[INFO] 2024-01-15 12:00:01 - 数据集 wells 包含 3 个项目
[INFO] 2024-01-15 12:00:05 - 算法执行完成

# 查看错误日志 (如有)
$ fraclab-sdk run tail f9e8d7c6 --stderr
```

> Workbench 提示：结果页面会展示本次运行的输出目录路径（含 `_logs` 日志），即使运行失败也能点开路径定位调试。

#### 6. 运行目录结构

执行完成后，`~/.fraclab/runs/<run_id>/` 目录结构:

```
f9e8d7c6/
├── run_meta.json              # 运行元数据
├── input/                     # 输入目录 (物化后的数据)
│   ├── manifest.json          # 输入清单 (含哈希)
│   ├── ds.json                # 运行数据规格 (重新索引)
│   ├── drs.json               # 算法 DRS
│   ├── params.json            # 用户参数
│   ├── run_context.json       # 运行上下文
│   └── data/                  # 数据目录
│       └── wells/
│           └── object.ndjson
└── output/                    # 输出目录
    ├── manifest.json          # 输出清单 ★ 核心结果文件
    ├── artifacts/             # 制品文件目录
    │   ├── summary.json
    │   └── report.pdf
    ├── _logs/                 # 日志目录
    │   ├── stdout.log         # 标准输出
    │   ├── stderr.log         # 标准错误
    │   ├── algorithm.log      # 算法日志 (DEBUG 级别)
    │   └── execute.json       # 执行元数据
```

#### 7. 输出 manifest.json 完整示例

```json
{
  "schemaVersion": "1.0",
  "run": {
    "runId": "f9e8d7c6",
    "algorithmId": "my-algorithm",
    "contractVersion": "1.0.0",
    "codeVersion": "1.0.0"
  },
  "status": "succeeded",
  "startedAt": "2024-01-15T12:00:00.000Z",
  "completedAt": "2024-01-15T12:00:05.123Z",
  "datasets": [
    {
      "datasetKey": "artifacts",
      "items": [
        {
          "itemKey": "accuracy",
          "artifact": {
            "artifactKey": "accuracy",
            "type": "scalar",
            "value": 0.95
          }
        },
        {
          "itemKey": "summary",
          "artifact": {
            "artifactKey": "summary",
            "type": "json",
            "mimeType": "application/json",
            "uri": "file:///Users/.../output/artifacts/summary.json"
          }
        }
      ]
    },
    {
      "datasetKey": "reports",
      "items": [
        {
          "itemKey": "report",
          "owner": { "wellId": "W001" },
          "artifact": {
            "artifactKey": "report",
            "type": "blob",
            "mimeType": "application/pdf",
            "uri": "file:///Users/.../output/artifacts/report.pdf"
          }
        }
      ]
    }
  ]
}
```

### 算法开发命令

#### 编译算法

```bash
# 编译算法工作区
fraclab-sdk algo compile ./my-algorithm --bundle /path/to/bundle

# 生成:
# - dist/params.schema.json (从 schema.inputspec:INPUT_SPEC)
# - dist/output_contract.json (从 schema.output_contract:OUTPUT_CONTRACT)
# - dist/ds.json (从 bundle 复制)
# - dist/drs.json (从 bundle 复制)
```

#### 初始化算法工作区

```bash
# 创建本地算法脚手架（默认写入 ~/.fraclab/workspace_algorithms）
fraclab-sdk algo init my-algorithm --code-version 0.1.0 --contract-version 1.0.0

# 常用可选参数
fraclab-sdk algo init my-algorithm \
  --name "My Algorithm" \
  --summary "Algorithm summary" \
  --author-name "Your Name" \
  --author-email "you@example.com" \
  --tag test --tag smoke
```

#### 导出算法包

```bash
# 导出为 zip 包
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip

# 自动编译后导出
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle
```

#### 导入算法

```bash
# 导入算法到 SDK 库
fraclab-sdk algo import ./my-algorithm.zip

# 列出已导入的算法
fraclab-sdk algo list
```

### 验证命令

```bash
# 验证 InputSpec
fraclab-sdk validate inputspec ./my-algorithm

# 验证 OutputContract
fraclab-sdk validate output-contract ./my-algorithm

# 验证 Bundle 完整性
fraclab-sdk validate bundle /path/to/bundle

# 验证运行输出清单
fraclab-sdk validate run-manifest /path/to/manifest.json --contract /path/to/contract.json
```

### 快照管理命令

```bash
# 导入数据快照
fraclab-sdk snapshot import /path/to/bundle

# 列出已导入快照
fraclab-sdk snapshot list
```

### 运行管理命令

```bash
# 创建运行
fraclab-sdk run create <snapshot_id> <algorithm_id> <version> --params params.json

# 执行运行
fraclab-sdk run exec <run_id> --timeout 300

# 查看日志
fraclab-sdk run tail <run_id>
fraclab-sdk run tail <run_id> --stderr
```

### 结果查看命令

```bash
# 列出运行产出的制品
fraclab-sdk results list <run_id>
```

### 调试模式

```bash
# 显示完整堆栈跟踪
fraclab-sdk --debug <command>
```

---

<a id="sdk-internal-modules"></a>
## 6. SDK 内部模块

以下模块供进阶使用或二次开发。

### SDKConfig - 配置管理

```python
from fraclab_sdk import SDKConfig

# 使用默认路径 (~/.fraclab)
config = SDKConfig()

# 显式指定路径
config = SDKConfig(sdk_home="/custom/path")

# 通过环境变量: export FRACLAB_SDK_HOME=/custom/path
config = SDKConfig()  # 自动读取
```

属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `sdk_home` | `Path` | SDK 根目录 |
| `snapshots_dir` | `Path` | 快照存储目录 |
| `algorithms_dir` | `Path` | 算法存储目录 |
| `runs_dir` | `Path` | 运行存储目录 |

### SnapshotLibrary - 快照管理

```python
from fraclab_sdk import SnapshotLibrary, SDKConfig

lib = SnapshotLibrary(SDKConfig())

# 导入快照
snapshot_id = lib.import_snapshot("/path/to/bundle")

# 列出快照
snapshots = lib.list_snapshots()

# 获取快照句柄
snapshot = lib.get_snapshot(snapshot_id)

# 删除快照
lib.delete_snapshot(snapshot_id)
```

### SnapshotHandle - 快照访问

```python
snapshot = lib.get_snapshot(snapshot_id)

# 属性
snapshot.directory    # Path: 快照目录
snapshot.manifest     # BundleManifest: 清单
snapshot.dataspec     # DataSpec: 数据规格
snapshot.drs          # DRS: 数据需求规格
snapshot.data_root    # Path: 数据目录

# 方法
datasets = snapshot.get_datasets()                      # 所有数据集
items = snapshot.get_items("wells")                     # 数据集的所有项目
data = snapshot.read_object_line("wells", 0)            # 读取 NDJSON 行
item_dir = snapshot.get_item_dir("frames", 0)           # 获取项目目录
parts = snapshot.read_frame_parts("frames", 0)          # Parquet 分片列表
```

### AlgorithmLibrary - 算法管理

```python
from fraclab_sdk import AlgorithmLibrary, SDKConfig

lib = AlgorithmLibrary(SDKConfig())

# 导入算法
algorithm_id, version = lib.import_algorithm("/path/to/algo.zip")

# 列出算法
algorithms = lib.list_algorithms()

# 获取算法句柄
algorithm = lib.get_algorithm(algorithm_id, version)

# 删除算法
lib.delete_algorithm(algorithm_id, version)
```

### AlgorithmHandle - 算法访问

```python
algorithm = lib.get_algorithm(algorithm_id, version)

# 属性
algorithm.directory      # Path: 算法目录
algorithm.manifest       # AlgorithmManifest: 清单
algorithm.drs            # DRS: 数据需求规格
algorithm.params_schema  # dict: 参数 JSON Schema
algorithm.algorithm_path # Path: main.py 路径
```

### SelectionModel - 数据选择

```python
from fraclab_sdk import SelectionModel

# 从快照和 DRS 创建选择模型
selection = SelectionModel.from_snapshot_and_drs(snapshot, algorithm.drs)

# 获取可选数据集
for ds in selection.get_selectable_datasets():
    print(f"{ds.dataset_key}: 共 {ds.total_items} 项, 基数要求: {ds.cardinality}")

# 设置选择
selection.set_selected("wells", [0, 1, 2])

# 获取当前选择
selected = selection.get_selected("wells")  # [0, 1, 2]

# 验证选择
errors = selection.validate()
if not errors:
    print("选择有效")

# 构建运行数据规格 (重新索引)
run_ds = selection.build_run_ds()

# 获取索引映射 (run_index -> snapshot_index)
mapping = selection.get_selection_mapping("wells")
```

基数规则：
- `"one"`: 必须恰好选择 1 个
- `"many"`: 必须至少选择 1 个
- `"zeroOrMany"`: 可以选择 0 个或多个

### RunManager - 运行管理

```python
from fraclab_sdk import RunManager, SDKConfig

mgr = RunManager(SDKConfig())

# 创建运行
run_id = mgr.create_run(
    snapshot_id=snapshot_id,
    algorithm_id=algorithm_id,
    algorithm_version=version,
    selection=selection,
    params={"threshold": 0.8}
)

# 执行运行
result = mgr.execute(run_id, timeout_s=300)
print(f"状态: {result.status}")
print(f"退出码: {result.exit_code}")

# 查询状态
status = mgr.get_run_status(run_id)

# 列出运行
runs = mgr.list_runs()

# 获取运行目录
run_dir = mgr.get_run_dir(run_id)

# 删除运行
mgr.delete_run(run_id)
```

### ResultReader - 结果读取

```python
from fraclab_sdk import ResultReader

reader = ResultReader(run_dir)

# 检查清单
if reader.has_manifest():
    manifest = reader.read_manifest()
    print(f"状态: {manifest.status}")

# 列出制品
artifacts = reader.list_artifacts()
for art in artifacts:
    print(f"{art.artifactKey}: {art.type}")

# 获取制品
artifact = reader.get_artifact("score")
path = reader.get_artifact_path("metrics")
data = reader.read_artifact_json("metrics")
value = reader.read_artifact_scalar("score")

# 读取日志
stdout = reader.read_stdout()
stderr = reader.read_stderr()
algo_log = reader.read_algorithm_log()
```

### Devkit - 开发工具

```python
from fraclab_sdk.devkit import (
    compile_algorithm,
    export_algorithm_package,
    validate_inputspec,
    validate_output_contract,
    validate_bundle,
    validate_run_manifest,
)

# 编译
result = compile_algorithm(
    workspace="/path/to/workspace",
    bundle_path="/path/to/bundle",
)

# 导出
result = export_algorithm_package(
    workspace="/path/to/workspace",
    output="/path/to/output.zip",
    auto_compile=True,
    bundle_path="/path/to/bundle",
)

# 验证
result = validate_inputspec("/path/to/workspace")
result = validate_output_contract("/path/to/workspace")
result = validate_bundle("/path/to/bundle")
result = validate_run_manifest(
    manifest_path="/path/to/manifest.json",
    contract_path="/path/to/contract.json"
)

if result.valid:
    print("验证通过")
else:
    for issue in result.errors:
        print(f"[{issue.code}] {issue.path}: {issue.message}")
```

---

<a id="data-models"></a>
## 7. 数据模型

### DRS (Data Requirement Specification)

算法对输入数据的需求定义。

```python
from fraclab_sdk.models import DRS

drs = DRS.model_validate_json(json_string)
dataset = drs.get_dataset("wells")
print(dataset.cardinality)  # "one", "many", "zeroOrMany"
```

### DataSpec

数据规格定义，描述快照中的数据集结构。

```python
from fraclab_sdk.models import DataSpec

ds = DataSpec.model_validate_json(json_string)
dataset = ds.get_dataset("wells")
keys = ds.get_dataset_keys()
```

### OutputContract

算法输出合约，声明算法产出的数据结构。

```python
from fraclab_sdk.models import OutputContract

contract = OutputContract.model_validate_json(json_string)
dataset = contract.get_dataset("results")
artifacts = contract.get_all_artifacts()
```

### RunOutputManifest

运行输出清单，记录算法执行的结果。

```python
from fraclab_sdk.models import RunOutputManifest

manifest = RunOutputManifest.model_validate_json(json_string)
artifact = manifest.get_artifact("score")
all_artifacts = manifest.list_all_artifacts()
```

---

<a id="error-handling"></a>
## 8. 错误处理

### 异常类型

```python
from fraclab_sdk.errors import (
    FraclabError,          # 基类
    SnapshotError,         # 快照相关
    HashMismatchError,     # 哈希不匹配
    PathTraversalError,    # 路径穿越攻击
    AlgorithmError,        # 算法相关
    SelectionError,        # 选择相关
    DatasetKeyError,       # 数据集键不存在
    CardinalityError,      # 基数验证失败
    MaterializeError,      # 物化错误
    RunError,              # 运行相关
    TimeoutError,          # 执行超时
    ResultError,           # 结果读取
    OutputContainmentError,# 输出路径逃逸
)
```

### 退出码

```python
from fraclab_sdk.errors import ExitCode

ExitCode.SUCCESS        # 0: 成功
ExitCode.GENERAL_ERROR  # 1: 一般错误
ExitCode.INPUT_ERROR    # 2: 输入/验证错误
ExitCode.RUN_FAILED     # 3: 运行失败
ExitCode.TIMEOUT        # 4: 超时
ExitCode.INTERNAL_ERROR # 5: 内部错误
```

### 错误处理示例

```python
from fraclab_sdk.errors import FraclabError, HashMismatchError, CardinalityError

try:
    snapshot_id = lib.import_snapshot(path)
except HashMismatchError as e:
    print(f"文件 {e.file_name} 哈希不匹配")
    print(f"预期: {e.expected}")
    print(f"实际: {e.actual}")
except FraclabError as e:
    print(f"SDK 错误 (退出码={e.exit_code}): {e}")
```

---

<a id="security-features"></a>
## 9. 安全特性

SDK 内置多项安全机制：

1. **路径穿越防护**: 导入时验证所有路径，拒绝 `..` 和绝对路径
2. **哈希验证**: 验证 ds.json 和 drs.json 的 SHA256 哈希
3. **输出隔离**: 算法只能写入指定的 output 目录
4. **进程隔离**: 算法在子进程中执行，有超时控制
5. **原子写入**: 使用 tmp + rename 确保文件完整性

---

<a id="complete-examples"></a>
## 10. 完整示例

### 算法开发完整流程

```python
# 1. 编写算法
# main.py
def run(ctx):
    dc = ctx.data_client
    aw = ctx.artifacts

    for key in dc.get_dataset_keys():
        for idx, obj in dc.iterate_objects(key):
            result = analyze(obj)
            aw.write_scalar(f"result_{key}_{idx}", result)

    aw.write_json("summary", {"completed": True})

# 2. 定义规格
# schema/inputspec.py
from pydantic import BaseModel, Field

class Params(BaseModel):
    threshold: float = Field(default=0.5)

INPUT_SPEC = Params

# 3. 编译并导出
# $ fraclab-sdk algo compile ./my-algo --bundle /path/to/bundle
# $ fraclab-sdk algo export ./my-algo ./my-algo.zip
```

### 使用 SDK 执行算法

```python
from fraclab_sdk import (
    SDKConfig,
    SnapshotLibrary,
    AlgorithmLibrary,
    SelectionModel,
    RunManager,
    ResultReader,
)

config = SDKConfig()

# 导入资源
snap_lib = SnapshotLibrary(config)
algo_lib = AlgorithmLibrary(config)
run_mgr = RunManager(config)

snapshot_id = snap_lib.import_snapshot("/path/to/bundle")
algorithm_id, version = algo_lib.import_algorithm("/path/to/algo.zip")

# 创建选择
snapshot = snap_lib.get_snapshot(snapshot_id)
algorithm = algo_lib.get_algorithm(algorithm_id, version)
selection = SelectionModel.from_snapshot_and_drs(snapshot, algorithm.drs)
selection.set_selected("wells", [0, 1, 2])

# 执行
run_id = run_mgr.create_run(
    snapshot_id, algorithm_id, version,
    selection, {"threshold": 0.8}
)
result = run_mgr.execute(run_id)

# 读取结果
reader = ResultReader(run_mgr.get_run_dir(run_id))
for art in reader.list_artifacts():
    print(f"{art.artifactKey}: {art.type}")
```

---

<a id="appendix-a-bundle-structure"></a>
## 11. 附录 A: Bundle 结构详解

> 本节面向平台开发者或需要排查导入问题的用户。普通算法开发者无需了解这些细节。

### 最小可用 Bundle 目录结构

```
my-bundle/
├── manifest.json          # 必须: 清单文件 (含哈希校验)
├── ds.json                # 必须: 数据规格 (DataSpec)
├── drs.json               # 必须: 数据需求规格 (DRS)
└── data/                  # 必须: 数据目录
    └── wells/             # 数据集目录 (datasetKey)
        └── object.ndjson  # NDJSON 格式数据文件
```

### manifest.json 完整示例

```json
{
  "bundleVersion": "1.0.0",
  "createdAtUs": 1706000000000000,
  "specFiles": {
    "dsPath": "ds.json",
    "drsPath": "drs.json",
    "dsSha256": "a1b2c3d4e5f6...(64位十六进制)",
    "drsSha256": "f6e5d4c3b2a1...(64位十六进制)"
  },
  "dataRoot": "data",
  "datasets": {
    "wells": {
      "layout": "object_ndjson_lines",
      "count": 3
    }
  }
}
```

**字段说明:**

| 字段 | 必填 | 说明 |
|------|------|------|
| `bundleVersion` | 是 | 固定值 `"1.0.0"` |
| `createdAtUs` | 是 | 创建时间 (微秒时间戳) |
| `specFiles.dsPath` | 是 | ds.json 相对路径，默认 `"ds.json"` |
| `specFiles.drsPath` | 是 | drs.json 相对路径，默认 `"drs.json"` |
| `specFiles.dsSha256` | 是 | ds.json 的 SHA256 哈希 |
| `specFiles.drsSha256` | 是 | drs.json 的 SHA256 哈希 |
| `dataRoot` | 是 | 数据目录相对路径，默认 `"data"` |
| `datasets` | 是 | 数据集清单，key 为 datasetKey |
| `datasets.*.layout` | 是 | `"object_ndjson_lines"` 或 `"frame_parquet_item_dirs"` |
| `datasets.*.count` | 是 | 数据集中的项目数量 |

### ds.json (DataSpec) 完整示例

```json
{
  "schemaVersion": "1.0.0",
  "datasets": [
    {
      "key": "wells",
      "resourceType": "well",
      "layout": "object_ndjson_lines",
      "items": [
        { "owner": { "wellId": "W001" } },
        { "owner": { "wellId": "W002" } },
        { "owner": { "wellId": "W003" } }
      ]
    }
  ]
}
```

**字段说明:**

| 字段 | 必填 | 说明 |
|------|------|------|
| `datasets[].key` | 是 | 数据集键名，对应 data/ 下的目录名 |
| `datasets[].resourceType` | 否 | 资源类型标识 |
| `datasets[].layout` | 是 | 数据布局，决定数据存储格式 |
| `datasets[].items` | 是 | 项目列表，每个元素对应一条数据 |
| `datasets[].items[].owner` | 否 | 所有者标识，包含 platformId/wellId/stageId |

### drs.json (DRS) 完整示例

```json
{
  "schemaVersion": "1.0.0",
  "datasets": [
    {
      "key": "wells",
      "resource": "well",
      "cardinality": "many",
      "description": "井数据集"
    }
  ]
}
```

**字段说明:**

| 字段 | 必填 | 说明 |
|------|------|------|
| `datasets[].key` | 是 | 数据集键名，必须与 ds.json 中的 key 匹配 |
| `datasets[].resource` | 否 | 资源类型 (别名 `resourceType`) |
| `datasets[].cardinality` | 是 | 基数要求: `"one"` / `"many"` / `"zeroOrMany"` |
| `datasets[].description` | 否 | 数据集描述 |

**基数规则:**
- `"one"`: 必须恰好选择 1 个项目
- `"many"`: 必须至少选择 1 个项目
- `"zeroOrMany"`: 可以选择 0 个或多个项目

### 数据目录布局

**布局 1: object_ndjson_lines (NDJSON 格式)**

```
data/
└── wells/
    ├── object.ndjson      # 必须: 每行一个 JSON 对象
    └── object.idx.u64     # 可选: 索引文件 (加速随机访问)
```

`object.ndjson` 示例 (每行一个 JSON 对象):
```
{"wellId": "W001", "name": "Well Alpha", "depth": 3000}
{"wellId": "W002", "name": "Well Beta", "depth": 3500}
{"wellId": "W003", "name": "Well Gamma", "depth": 2800}
```

**布局 2: frame_parquet_item_dirs (Parquet 格式)**

```
data/
└── frames/
    └── parquet/
        ├── item-00000/    # 第 0 个项目
        │   └── data.parquet
        ├── item-00001/    # 第 1 个项目
        │   └── data.parquet
        └── item-00002/    # 第 2 个项目
            └── data.parquet
```

目录命名规则: `item-{index:05d}` (5 位数字，前导零填充)

### 生成哈希值

使用 Python 计算 SHA256 哈希:

```python
import hashlib
from pathlib import Path

def compute_sha256(file_path: str) -> str:
    content = Path(file_path).read_bytes()
    return hashlib.sha256(content).hexdigest()

# 示例
ds_hash = compute_sha256("ds.json")
drs_hash = compute_sha256("drs.json")
print(f"dsSha256: {ds_hash}")
print(f"drsSha256: {drs_hash}")
```
