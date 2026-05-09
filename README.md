# Sholar Agent in LangGraph

这是一个本地运行的学术论文阅读 agent 骨架，项目结构以 LangGraph 为中心。agent 启动后会自动扫描论文、生成缺失摘要、展示待读摘要；用户不是控制流调度者，而是 graph 中的 human-in-the-loop 节点。

核心是一张自动执行的图：

- `initialize_memory`: 初始化长期记忆数据库
- `scan_library`: 递归扫描 PDF 文件夹并登记论文
- `select_unsummarized_paper`: 从数据库选择摘要未生成的论文
- `generate_summary`: 读取 PDF 原文并生成摘要、关键词、分类
- `load_review_queue`: 加载摘要已生成但用户未读的论文
- `human_summary_review`: 暂停给用户看摘要，用户只选择“深度分析”或“跳过”
- `record_summary_decision`: 记录用户选择
- `prepare_deep_analysis_context`: 读取原文和同分类历史笔记
- `draft_deep_analysis_note`: 生成深度分析笔记初稿
- `human_note_review`: 暂停给用户看笔记初稿，用户提出修改意见或确认终稿
- `revise_deep_analysis_note`: 根据用户反馈修改笔记初稿，并再次回到 human 节点
- `save_final_note`: 保存终稿并更新数据库

LangGraph 入口在 [langgraph.json](/home/hjr/sholarAgentinLangGraph/langgraph.json) 中声明：

```json
{
  "dependencies": ["."],
  "graphs": {
    "research_agent": "./sholar_agent/agent.py:graph"
  },
  "env": "./.env"
}
```

## 安装

```bash
cd /home/hjr/sholarAgentinLangGraph
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`python -m venv .venv` 会创建项目自己的 Python 虚拟环境。

`source .venv/bin/activate` 会启用这个虚拟环境。

`pip install -e ".[dev]"` 会用可编辑模式安装当前项目，并安装测试、LangGraph CLI 等开发工具。

## 配置

```bash
cp .env.example .env
```

核心配置：

- `SHOLAR_DB_PATH`: SQLite 数据库路径，默认 `./data/sholar_agent.sqlite3`
- `SHOLAR_PDF_ROOT`: 默认递归扫描 PDF 的根目录，默认 `./papers`
- `SHOLAR_LLM_PROVIDER`: LLM 适配器名，当前为 `placeholder`
- `SHOLAR_LLM_MODEL`: 模型名，占位字段

## 启动

```bash
langgraph dev
```

LangGraph 会读取 `langgraph.json`，加载 `research_agent` 这张图。你可以在 LangGraph Studio 或 SDK 中启动 graph。graph 启动后会自动执行：

1. 初始化数据库
2. 扫描 PDF 目录
3. 为所有摘要未生成的论文生成摘要
4. 逐篇展示“摘要已生成且用户未读”的论文
5. 用户选择 `d` 深度分析或 `s` 跳过
6. 如果深度分析，agent 生成笔记初稿并暂停等待用户修改或确认
7. 确认后保存终稿，然后继续处理下一篇待读论文

用户介入通过 LangGraph `interrupt()` 完成：`human_summary_review` 返回 `deep_analysis` 或 `skip`，`human_note_review` 返回 `revise` 或 `confirm`。

## 数据库设计

主表：`papers`

- `title`: 论文标题，主键，作为唯一 id
- `conference`: 论文所属会议
- `publication_time`: 论文发表时间
- `summary_generated`: 论文是否生成摘要
- `user_read`: 用户是否已读摘要
- `deep_analyzed`: 是否深度分析
- `note_path`: 论文笔记文件位置
- `pdf_path`: 论文 PDF 文件位置
- `summary`: 摘要内容
- `keywords_json`: 关键词列表 JSON
- `categories_json`: 分类列表 JSON
- `created_at`, `updated_at`: 本地记录时间

辅助表：`deep_analysis_sessions` 记录深度分析初稿、确认状态和终稿路径。

## 后续你主要实现哪里

1. 在 [sholar_agent/utils/tools.py](/home/hjr/sholarAgentinLangGraph/sholar_agent/utils/tools.py) 替换 `PlaceholderLLMClient`。
2. 在 [sholar_agent/utils/nodes.py](/home/hjr/sholarAgentinLangGraph/sholar_agent/utils/nodes.py) 细化每个自动节点和 human-in-the-loop 节点。
3. 在 [sholar_agent/utils/state.py](/home/hjr/sholarAgentinLangGraph/sholar_agent/utils/state.py) 扩展 agent 的共享状态。
4. 在 [sholar_agent/utils/tools.py](/home/hjr/sholarAgentinLangGraph/sholar_agent/utils/tools.py) 调整摘要和笔记模板。
