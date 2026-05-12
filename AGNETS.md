# AGNETS

这个文件是给后续 Codex 读项目、改项目用的导航。

## 项目目标

这个仓库是一个基于 LangGraph 的论文阅读 agent，目标流程是：

1. 扫描 `papers/` 下的 PDF
2. 直接把原始 PDF 作为附件发送给大模型，生成结构化摘要
3. 把已摘要但未阅读的论文交给人类确认
4. 需要时生成深度笔记初稿
5. 接收人类修改意见并保存终稿

## 关键入口

- `langgraph.json`：LangGraph 读取 graph 的入口配置
- `scholar_agent/agent.py`：图定义、路由、graph 导出
- `scholar_agent/utils/nodes.py`：所有节点逻辑
- `scholar_agent/utils/tools.py`：数据库、LLM、PDF 附件构建、文件读写等基础工具
- `scholar_agent/utils/state.py`：状态与数据结构定义

## 目录职责

### `scholar_agent/agent.py`

负责把各个节点串成图，并导出：

- `graph`：给 `langgraph dev` / LangGraph API 用
- `local_graph`：给本地脚本调试用，带 `InMemorySaver`

这里的重点是路由函数：

- `route_after_select_unsummarized`
- `route_after_review_queue`
- `route_after_summary_decision`
- `route_after_note_review`

如果要改流程走向，优先改这里。

### `scholar_agent/utils/state.py`

定义项目中的核心数据结构：

- `Paper`
- `SummaryResult`
- `PaperRecord`
- `ResearchAgentState`

如果新增节点字段、状态字段、返回值字段，先改这里。

### `scholar_agent/utils/nodes.py`

所有 graph 节点都在这里：

- `initialize_memory_node`
- `scan_library_node`
- `select_unsummarized_paper_node`
- `generate_summary_node`
- `load_review_queue_node`
- `human_summary_review_node`
- `record_summary_decision_node`
- `prepare_deep_analysis_context_node`
- `draft_deep_analysis_note_node`
- `human_note_review_node`
- `revise_deep_analysis_note_node`
- `save_final_note_node`

这里负责业务编排，不建议塞太多底层 I/O 细节。

当前 PDF 流程是：

- `generate_summary_node` 直接把 `build_pdf_attachment(paper.pdf_path)` 传给 LLM
- `prepare_deep_analysis_context_node` 先筛选同分类历史笔记，再按关键词命中数排序，保留前 5 篇
- `draft_deep_analysis_note_node` 也直接把 PDF 附件传给 LLM，并在首次生成后立即把草稿写到 `*.notes.md`
- 本地主要负责扫描文件、提取标题、构造附件和保存结果

当前两个 human 节点的输入约定：

- `human_summary_review_node`
  - 接受 `deep_analysis` / `skip`
  - 也接受 `0` / `1`
  - `0 -> deep_analysis`
  - `1 -> skip`
- `human_note_review_node`
  - 展示的是当前已落盘的笔记内容
  - 把返回值按字符串处理
  - 第一个词是 `confirm` 就走保存
  - 其他情况都视为用户问题或修改请求，先回答问题，再基于回答修订笔记并覆盖原草稿文件

### `scholar_agent/utils/tools.py`

这里放所有基础设施和适配层：

- `Settings`：环境变量配置
- `get_settings`
- `get_repository`
- `get_llm`
- `PaperRepository`
- `PlaceholderLLMClient`
- `DeepSeekLLMClient`
- PDF 扫描、标题提取、base64 附件构建函数
- 文件写入函数

如果要换数据库、换 LLM、改 PDF 附件发送逻辑，优先改这里。

注意：当前 `llm_model` 最好显式配置，不再依赖旧版默认兜底模型名。

### `scholar_agent/config.py`

放可直接调整的提示词模板：

- `SUMMARY_TEMPLATE`
- `DEEP_ANALYSIS_TEMPLATE`

如果想改摘要风格或 deep_analysis 输出格式，优先改这里。

### `langgraph.json`

声明 LangGraph 要加载哪个 graph：

- `research_agent -> ./scholar_agent/agent.py:graph`

### `README.md`

面向人类的项目说明。

### `todo.md`

临时任务清单，可能带有未完成的改动方向。

## 运行数据

- `papers/`：PDF 文件目录
- `data/`：SQLite 数据库等本地持久化
- `./papers/*.notes.md`：生成的终稿笔记

## 当前运行方式

- 开发调试：`langgraph dev`
- 运行时入口：`graph`
- 本地调试入口：`local_graph`

注意：

- `graph` 不要挂自定义 `InMemorySaver`，LangGraph API 会自己接管持久化
- `local_graph` 才是本地手动调试时用的内存 checkpointer 图

## 修改建议

### 如果你要改流程

优先看：

- `scholar_agent/agent.py`
- `scholar_agent/utils/nodes.py`

### 如果你要改数据结构

优先看：

- `scholar_agent/utils/state.py`
- `scholar_agent/utils/tools.py`

### 如果你要改用户交互

优先看：

- `human_summary_review_node`
- `human_note_review_node`

### 如果你要改存储

优先看：

- `PaperRepository`
- `write_final_note`
- `read_related_notes`

## 已知约定

- 当前项目名是 `sholar-agent-in-langgraph`
- 这是一个 v0.1 阶段的项目，先保证能跑，再逐步增强
- 这里的 `AGNETS.md` 是项目导航文件，后续 Codex 可以先读它再进入具体实现
