# MiMo 智能路线规划系统

基于 LangGraph 多智能体 + MiMo 大模型的智能路线规划系统。系统采用 **Supervisor 多 Agent 协同架构**，由 Supervisor 主管 Agent 协调 IntentAgent（意图分析）、PlanningAgent（路线规划）、ExplanationAgent（解释生成）三个专业子 Agent 协同完成路线规划全流程。

支持城市：北京、上海、广州、深圳、成都、杭州、武汉、西安、重庆、南京、天津、苏州、长沙、青岛、郑州、厦门、昆明、大连、三亚、丽江。

## 核心特性

- **多智能体协同**：Supervisor 主管模式，3 个专业 Agent（意图分析、路线规划、解释生成）协同工作
- **LLM 驱动**：使用小米 MiMo 大模型进行意图解析、路线规划和解释生成
- **精美前端**：深色毛玻璃主题 Web 界面，实时展示 Agent 工作流、意图解析、路线卡片和推荐理由
- **流式输出**：支持 SSE 流式响应，实时展示规划进度和解释内容
- **商圈分片索引**：20 城市 × 150+ 热门商圈，用户提到"春熙路"会优先命中对应商圈 shard，检索更快更精准
- **多方案推荐**：每次生成 3 条差异化路线（综合最优、少排队优先、低预算优先）
- **本地 POI 数据**：50 万条 POI 作为知识库，运行时主入口为 `SQLite + 城市/商圈/区县查询`
- **自动识别链路**：城市、商圈、默认 alias、district、zone center 优先从 SQLite 读取，旧 manifest 和手写常量仅作为补丁层
- **二次规划**：支持自然语言反馈重新调整路线
- **耗时统计**：API 响应包含 `elapsed` 字段（秒），方便性能观测

## 技术架构

```
用户输入（自然语言）
    ↓
Web 前端 (static/index.html)
    ↓ SSE 流式请求
FastAPI 接口 (app.py)
    ↓
Supervisor 主管 Agent (core/agent.py)
    ↓ 确定性路由（基于消息历史）
    ├── IntentAgent (意图分析专家)
    │   ├── parse_intent → 意图解析
    │   ├── get_user_profile → 用户画像
    │   └── list_supported_cities → 城市查询
    │
    ├── PlanningAgent (路线规划专家)
    │   ├── retrieve_pois → POI 检索
    │   ├── plan_routes → 路线规划
    │   └── replan_routes → 路线重规划
    │
    └── ExplanationAgent (解释生成专家)
        └── explain_routes → 路线解释生成
```

## 目录结构

```text
rounter-main/
├── app.py                            # FastAPI 入口，API 路由定义
├── cli.py                            # 命令行测试工具
├── generate_poi_data.py              # 数据生成脚本（50万条 POI，生成 SQLite + JSON / 分片兼容产物）
├── .env.example                      # 环境变量配置示例
├── static/
│   └── index.html                    # Web 前端（深色毛玻璃主题，SSE 实时展示）
├── data/
│   ├── poi_data_500k.db              # 运行时主数据源（SQLite）
│   ├── poi_data_500k.json            # 兼容旧加载路径的单文件 JSON
│   ├── poi_data_500k_partitioned/
│   │   ├── manifest.json             # 兼容层：城市/商圈/区县分片索引 + zone center
│   │   └── cities/
│   │       └── <城市>/
│   │           ├── all.json          # 城市级分片
│   │           ├── zones/            # 商圈级分片
│   │           └── districts/        # 区县级分片
│   └── user_profiles.json            # 用户画像数据（6种类型）
├── core/
│   ├── agent.py                      # Supervisor 多 Agent 编排（主管路由、子 Agent 调度）
│   ├── sub_agents.py                 # 子 Agent 定义（IntentAgent、PlanningAgent、ExplanationAgent）
│   ├── agent_tools.py                # LangChain 工具定义（按子 Agent 分组的 7 个 Tool）
│   ├── mimo_client.py                # MiMo API 统一客户端（OpenAI/Anthropic 协议，支持流式）
│   ├── intent_parser.py              # LLM 意图解析 + 规则回退
│   ├── poi_artifact_store.py         # POI 分片 artifact loader（monolithic / partitioned）
│   ├── zone_catalog.py               # SQLite / manifest 驱动的城市商圈识别目录
│   ├── poi_retriever.py              # POI 召回与评分
│   ├── ugc_analyzer.py               # UGC 评论分析
│   ├── scorer.py                     # 评分计算
│   ├── route_optimizer.py            # 本地路线生成 + LLM 兜底
│   ├── replanner.py                  # 反馈理解 + 重新规划
│   ├── explanation.py                # LLM 解释生成 + 模板回退
│   └── preference.py                 # 偏好匹配工具
├── models/
│   ├── schemas.py                    # 数据模型（Pydantic）
│   └── config.py                     # 配置文件（含 MiMo API 配置）
├── utils/
│   ├── geo.py                        # 地理距离计算
│   └── time_utils.py                 # 时间工具
├── tests/
│   └── test_e2e.py                   # 端到端测试 / manifest 识别覆盖测试
└── requirements.txt
```

## 环境准备

### 1. 创建 Python 环境

```bash
conda create -n ai_rounter python=3.12 -y
conda activate ai_rounter
pip install -r requirements.txt
```

### 2. 配置 LLM API

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
copy .env.example .env
```

编辑 `.env` 文件，选择一个模型：

```env
# Provider: 'openai' for most models, 'anthropic' for MiMo Anthropic protocol
ROUTE_PLANNER_LLM_PROVIDER=openai

# API Key
ROUTE_PLANNER_MIMO_API_KEY=your-api-key-here

# --- DeepSeek ---
ROUTE_PLANNER_MIMO_BASE_URL=https://api.deepseek.com
ROUTE_PLANNER_MIMO_MODEL=deepseek-chat

# --- LongCat (美团) ---
# ROUTE_PLANNER_MIMO_BASE_URL=https://api.longcat.ai/v1
# ROUTE_PLANNER_MIMO_MODEL=longcat-chat

# --- MiMo (OpenAI 协议) ---
# ROUTE_PLANNER_MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
# ROUTE_PLANNER_MIMO_MODEL=mimo-v2.5-pro

# --- MiMo (Anthropic 协议) ---
# ROUTE_PLANNER_LLM_PROVIDER=anthropic
# ROUTE_PLANNER_MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/anthropic
# ROUTE_PLANNER_MIMO_MODEL=mimo-v2.5-pro

# --- 硅基流动 ---
# ROUTE_PLANNER_MIMO_BASE_URL=https://api.siliconflow.cn/v1
# ROUTE_PLANNER_MIMO_MODEL=XiaomiMiMo/MiMo-7B-RL
```

### 3. 验证 API 连通性

```bash
python verify_api.py
```

输出 `[PASS]` 表示配置正确，可以开始使用。

### 4. 生成数据（首次使用）

```bash
python generate_poi_data.py
```

生成文件位于 `data/` 目录：
- `poi_data_500k.db`：**当前运行时主数据源**，包含 `cities / zones / districts / pois` 四张表
- `poi_data_500k.json`：兼容旧加载路径的单文件 JSON（20 个城市，500,000 条 POI，按城市分组）
- `poi_data_500k_partitioned/manifest.json`：兼容层索引，记录所有城市、商圈、区县分片以及商圈中心点
- `poi_data_500k_partitioned/cities/<城市>/all.json`：城市级 POI 分片
- `poi_data_500k_partitioned/cities/<城市>/zones/<商圈>.json`：商圈级 POI 分片
- `poi_data_500k_partitioned/cities/<城市>/districts/<区县>.json`：区县级 POI 分片

系统现在优先使用 **SQLite** 作为运行时数据源。用户说“春熙路”“曾厝垵”时，会优先命中对应商圈查询，而不是先解析整城 JSON。分片 JSON 与 manifest 继续保留，主要用于兼容、回滚和离线比对。

```json
{
  "format": "partitioned_json_v1",
  "cities": {
    "成都": {
      "all_file": "cities/成都/all.json",
      "zones": {
        "春熙路商圈": {
          "file": "cities/成都/zones/春熙路商圈.json",
          "center": [104.081, 30.657],
          "district": "锦江区"
        }
      }
    }
  }
}
```

每个城市定义了 5-15 个热门商圈（共 150+ 个），约 70% 的 POI 会落入商圈，30% 散落在城市其他区域。商圈识别链路现在优先从 SQLite 中的 `zones` / `districts` 元数据自动读取城市、商圈、中心点与默认 alias（如去掉“商圈”后缀）；manifest 仅作为兼容层与回滚参考。

## 命令行工具

### 查看用户画像

```bash
python cli.py profiles
```

### 规划路线

```bash
# 使用默认用户（u001 文艺慢逛型），默认流式输出
python cli.py plan "下午从成都春熙路出发，想吃火锅、拍照，不想排队，预算300，晚上9点前结束"

# 指定用户画像
python cli.py plan "下午想吃火锅，预算300" --user u005
python cli.py plan "带孩子出去玩，预算300" --user u002
python cli.py plan "晚上和朋友喝酒看夜景" --user u004

# 禁用流式输出，一次性返回完整 JSON 结果
python cli.py plan "下午想吃火锅，预算300" --no-stream
```

#### 更多规划测试示例

下面这些示例适合你直接复制测试，覆盖多城市、多商圈、多预算、多时段、多强偏好与不同人群场景。

```bash
# 成都：火锅 + 拍照 + 不排队
python cli.py plan "在成都，下午从春熙路出发，想吃火锅、拍照，不想排队，预算300，晚上9点前结束"

# 成都：雨天 + 室内 + 少走路
python cli.py plan "在成都，今天下雨，下午想在太古里附近逛一逛，尽量安排室内，少走路，预算250"

# 成都：奶茶 + 小吃 + 低预算
python cli.py plan "在成都，下午从成都建设路出发，想喝奶茶、吃小吃，预算120，少走路，别太累"

# 北京：晚饭 + 夜景 + 朋友聚会
python cli.py plan "在北京，晚上6点从三里屯出发，和朋友想吃点好的，再看看夜景，预算400"

# 北京：白天逛展 + 咖啡
python cli.py plan "在北京，周末下午从中关村出发，想看展、逛书店、喝咖啡，预算280"

# 上海：咖啡 + 城市漫步 + 文艺路线
python cli.py plan "在上海，周末下午从新天地出发，想喝咖啡、逛书店、拍照，节奏轻松一点，预算300"

# 上海：夜景 + 晚饭 + 少走路
python cli.py plan "在上海，傍晚从陆家嘴出发，想先吃晚饭再看夜景，少走路，预算420"

# 广州：低预算 + 奶茶 + 小吃
python cli.py plan "在广州，下午在北京路附近想喝奶茶、吃小吃，预算100，少走路"

# 广州：亲子 + 室内 + 雨天友好
python cli.py plan "在广州，下雨天想带孩子在天河城附近玩，尽量室内，预算300"

# 深圳：商场室内路线
python cli.py plan "在深圳，天气太热了，下午想在海岸城附近安排室内路线，带点咖啡休息，预算350"

# 深圳：下班后轻松约会
python cli.py plan "在深圳，晚上从福田CBD出发，情侣约会，想先吃饭再找个安静地方坐坐，预算380"

# 杭州：白天公园 / 西湖路线
python cli.py plan "在杭州，上午从西湖出发，想白天看看风景、拍照、喝咖啡，预算280"

# 杭州：傍晚散步 + 晚饭
python cli.py plan "在杭州，傍晚从湖滨出发，想慢慢散步，顺便吃个晚饭，预算260"

# 武汉：傍晚开始的夜生活路线
python cli.py plan "在武汉，傍晚从楚河汉街出发，想吃饭、散步、看夜景，预算320"

# 西安：游客型需求，先逛再吃
python cli.py plan "在西安，下午在大雁塔附近玩，想先逛逛再吃点本地小吃，预算200"

# 重庆：低预算 + 小吃 + 夜景
python cli.py plan "在重庆，晚上从解放碑出发，想吃小吃、看看夜景，预算150"

# 南京：情侣约会
python cli.py plan "在南京，周末下午从新街口出发，情侣约会，想吃饭、喝咖啡、散步，预算400"

# 苏州：古城 + 茶饮 + 轻松路线
python cli.py plan "在苏州，下午从观前街出发，想喝奶茶、逛逛古城，少走路，预算180"

# 厦门：游客打卡 + 海边氛围
python cli.py plan "在厦门，下午想在曾厝垵附近玩，想拍照、喝奶茶、慢慢逛，预算220"

# 昆明：白天翠湖休闲路线
python cli.py plan "在昆明，上午从翠湖出发，想白天散步、喝咖啡、看看展，预算260"

# 大连：海边 + 晚饭 + 夜景
python cli.py plan "在大连，傍晚从星海广场出发，想散步、吃饭、看看海边夜景，预算360"

# 三亚：傍晚海边路线
python cli.py plan "在三亚，傍晚想在大东海附近散步、吃饭、看海，预算350"

# 丽江：古城慢逛
python cli.py plan "在丽江，下午从大研古城出发，想慢慢逛、拍照、喝点东西，预算200"
```

### 重新规划

```bash
python cli.py replan "太贵了，控制在100以内"
python cli.py replan "不要火锅了，换成小吃"
python cli.py replan "下雨了，安排室内"
python cli.py replan "晚点出发，下午3点开始"
python cli.py replan "少走路，别太累"
```

#### 更多重新规划测试示例

这些反馈适合在你先跑出一条路线之后继续测试系统的重规划能力。

```bash
# 预算调整
python cli.py replan "太贵了，控制在150以内"
python cli.py replan "预算再低一点，100以内最好"
python cli.py replan "这条路线可以贵一点，预算提高到500"

# 餐饮偏好切换
python cli.py replan "不要火锅了，换成小吃"
python cli.py replan "不想吃正餐了，改成喝咖啡和奶茶"
python cli.py replan "把餐饮换成更适合拍照的店"

# 时段调整
python cli.py replan "晚点出发，下午3点开始"
python cli.py replan "我想早点结束，晚上8点前回去"
python cli.py replan "晚饭安排到6点半以后更合适"

# 天气 / 室内外调整
python cli.py replan "下雨了，安排室内"
python cli.py replan "天气太热了，尽量少在室外走"
python cli.py replan "太阳下山以后再去看夜景"

# 体力与节奏调整
python cli.py replan "少走路，别太累"
python cli.py replan "节奏放慢一点，不要排太满"
python cli.py replan "我今天状态不错，可以多玩一个点"

# 人群 / 场景变化
python cli.py replan "现在变成两个人一起去了，适合约会一点"
python cli.py replan "临时带了孩子，换成亲子友好的路线"
python cli.py replan "朋友加入了，想热闹一点"

# 强偏好补充
python cli.py replan "我还是很想吃火锅，优先安排火锅"
python cli.py replan "顺路买杯奶茶吧"
python cli.py replan "想加一个适合拍照的点"
```

### 自己构造测试 query 的方向

如果你想自己随机构造需求，下面这些维度组合起来会比较容易覆盖系统边界：

- 城市 / 商圈：`春熙路 / 三里屯 / 新天地 / 西湖 / 解放碑 / 大雁塔 / 曾厝垵`
- 强偏好：`火锅 / 小吃 / 咖啡 / 奶茶 / 室内 / 夜景`
- 时间条件：`上午 / 下午 / 傍晚 / 晚上 / 9点前结束 / 晚饭后再去`
- 体力条件：`少走路 / 别太累 / 节奏轻松`
- 天气条件：`下雨 / 太热 / 想室内`
- 人群场景：`一个人 / 情侣 / 朋友 / 带孩子 / 家庭`
- 预算条件：`100 / 150 / 300 / 500`
- 反馈方向：`太贵了 / 不要这个 / 换成别的 / 晚点出发 / 早点结束 / 再加一个点`

### 查看数据概览

```bash
python cli.py pois
```

## Web 前端

启动服务后浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000) 即可使用。

功能：
- 输入出行需求 + 选择用户画像，点击"开始规划"
- 实时展示 Agent 工作流动画（IntentAgent → PlanningAgent → ExplanationAgent）
- 意图卡片：城市、商圈、时段、预算、偏好标签
- 3 张路线卡片：费用、交通时间、停留时间、途经点列表
- 路线推荐理由（LLM 生成，打字机效果）
- 二次规划：输入反馈后自动重新规划

## API 接口

启动 FastAPI 服务：

```bash
uvicorn app:app --reload
```

启动后访问 Swagger UI：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 生成路线

```bash
curl -X POST http://127.0.0.1:8000/plan \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","query":"下午从春熙路出发，想吃火锅、拍照，预算300"}'
```

### 生成路线（流式 SSE）

```bash
curl -X POST http://127.0.0.1:8000/plan/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","query":"下午从春熙路出发，想吃火锅、拍照，预算300"}'
```

流式响应事件类型：
- `progress`：规划进度（正在解析意图、检索地点、规划路线等）
- `intent`：解析后的用户意图
- `routes`：规划完成的路线数据
- `explanation_chunk`：解释文本的流式片段
- `done`：完成事件，包含耗时和元数据

### 重新规划

```bash
curl -X POST http://127.0.0.1:8000/replan \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","previous_intent":{...},"feedback":"太贵了，控制在150以内"}'
```

### 重新规划（流式 SSE）

```bash
curl -X POST http://127.0.0.1:8000/replan/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","previous_intent":{...},"feedback":"太贵了，控制在150以内"}'
```

## Agent 架构

系统采用 **Supervisor 多 Agent 协同架构**：

1. **Supervisor 层** (`core/agent.py`)：主管 Agent，使用 LangGraph StateGraph 编排子 Agent，基于确定性路由（检查 ToolMessage 历史）按顺序调度子 Agent
2. **子 Agent 层** (`core/sub_agents.py`)：3 个专业子 Agent，各自拥有独立工具和系统提示
   - **IntentAgent**：意图分析专家，负责解析需求和获取用户画像
   - **PlanningAgent**：路线规划专家，负责检索 POI 和规划路线
   - **ExplanationAgent**：解释生成专家，负责生成路线友好解释
3. **工具层** (`core/agent_tools.py`)：7 个 LangChain Tool，按子 Agent 分组
4. **核心模块层** (`core/` 下各模块)：意图解析、POI 检索、路线优化、解释生成等

### 依赖

```
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
```

## 运行测试

```bash
pytest tests/ -v
```

## MiMo API 支持平台

| 平台 | Base URL | 说明 |
|------|----------|------|
| 硅基流动 SiliconFlow | `https://api.siliconflow.cn/v1` | 国内主流平台，注册即用 |
| 小米官方 | 查看 developer.mi.com | 官方 API 服务 |
| 本地 vLLM | `http://localhost:8000/v1` | 自部署，需 GPU |
| 本地 Ollama | `http://localhost:11434/v1` | 本地部署，简单易用 |

## 设计原则

1. **多 Agent 协同**：Supervisor 主管模式，3 个专业子 Agent 各司其职、协同完成任务
2. **LLM 优先**：核心流程优先使用 MiMo LLM 进行智能分析
3. **数据驱动**：本地 50 万条 POI 数据作为规划与召回的知识基础
4. **结构化输出**：通过 prompt engineering 约束 LLM 输出 JSON，确保结果可解析
5. **工具化封装**：核心模块以 LangChain Tool 形式暴露，Agent 可灵活组合调用
6. **Supervisor 编排**：主管 Agent 基于确定性路由（检查 ToolMessage 历史）按顺序调度子 Agent，保证稳定可靠的执行流程

## 性能优化

数据加载现在采用 **SQLite + 自动商圈识别 + 缓存策略**，避免每次请求重复解析单一大 JSON：

| 优化项 | 说明 |
|--------|------|
| SQLite 主数据源 | 运行时默认读取 `poi_data_500k.db`，无需先解析 600MB+ 全量 POI JSON |
| 按城市查询 | `cities` / `pois` 表支持按城市快速查询，不需要先展开全部 POI |
| 按商圈查询 | 用户提到“春熙路”“曾厝垵”等商圈名时，优先走 `zone:*` 查询 |
| 按区县查询 | `district:*` 作用域优先走 `district` 查询 |
| 自动识别链路 | 城市、商圈、默认 alias、商圈中心点优先从 SQLite 的 `zones` 元数据读取，手写常量只作为补丁层 |
| nearby 自动化 | `nearby` 逻辑优先使用 SQLite 中 zone center 计算 |
| 运行时缓存 | 已加载的城市/商圈查询结果、已验证 POI、用户画像都会缓存 |
| 兼容层保留 | `poi_data_500k.json` 与 `poi_data_500k_partitioned/` 继续保留，便于回滚、离线比对和兼容旧路径 |

API 响应中 `elapsed` 字段记录了每次规划的总耗时（秒），位于 `explanation` 之后。
