from langgraph.graph import Graph, StateGraph, START, END
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.messages import AIMessage, HumanMessage
# from langchain_openai import ChatOpenAI
from .no_think_chat_model import QwenNoThinkModel
from eventrag.tools import KnowledgeGraphTool, KnowledgeGraphEdgeTool, KeywordsQueryTool, EventGraphTool
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolInvocation
import logging
import subprocess
import tempfile
import os
import json
import re
from .prompt import PROMPTS
from .graph_state import MainAgentState, MAX_ITERATIONS, create_retry_decorator
from .common_branch import create_query_planner, execute_queries_wrapper, create_result_analyzer, create_reflection_analyzer
from .event_branch import create_event_query_executor, create_event_summarizer, create_event_aggregator, create_event_reflection_analyzer
logger = logging.getLogger(__name__)


def create_tool_executor(
    knowledge_graph_inst,
    entities_vdb,
    relationships_vdb,
    text_chunks_db,
    query_param
):
    tools = [
        EventGraphTool(
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            text_chunks_db=text_chunks_db,
            query_param=query_param
        ),
        KeywordsQueryTool(
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            text_chunks_db=text_chunks_db,
            query_param=query_param
        )
    ]
    return ToolExecutor(tools)

def create_keyword_generator(llm):
    template = PROMPTS["keywords_extraction"]
    
    @create_retry_decorator()
    async def generate_keywords(state: MainAgentState) -> MainAgentState:
        messages = state["messages"]
        question = messages[-1].content
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
        response = await llm.ainvoke(template.format(query=question, examples=examples, language=PROMPTS["DEFAULT_LANGUAGE"]))
        result = response.content
        logger.info(f"keywords_extraction result: {result}")
        try:
            if "{{" in result:
                result = result.replace("{{", "{").replace("}}", "}")
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                result = match.group(0)
                keywords_data = json.loads(result)

                hl_keywords = keywords_data.get("high_level_keywords", [])
                ll_keywords = keywords_data.get("low_level_keywords", [])
            else:
                logger.error("No JSON-like structure found in the result.")
                return PROMPTS["fail_response"]

        # Handle parsing error
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e} {result}")
            return PROMPTS["fail_response"]

        if hl_keywords == [] and ll_keywords == []:
            logger.warning("low_level_keywords and high_level_keywords is empty")
            return PROMPTS["fail_response"]

        state["query_results"]["keywords"] = f"{', '.join(ll_keywords)} || {', '.join(hl_keywords)}"
        state["next_keywords"] = ll_keywords + hl_keywords
        
        logger.info(f"ll_keywords: {ll_keywords}, hl_keywords: {hl_keywords}")
        return state
    
    return generate_keywords

def create_final_answer_generator(llm):
    template = """
    You are a helpful assistant responding to questions about data in the tables provided.

    ---Goal---

    Generate a response that synthesizes:
    1. Knowledge graph analysis: {analysis}
    2. Event-based analysis: {event_analysis}

    Original question: {question}

    Ensure the response:
    - Integrates both structural knowledge and event-based information
    - Highlights temporal and causal relationships where relevant
    - Maintains logical flow and coherence
    
    Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown. Keep the same language as `Original question`.
    """
    @create_retry_decorator()
    async def generate_final_answer(state: MainAgentState) -> MainAgentState:
        question = state["messages"][-1].content
        analysis = state["query_results"].get("analysis", "")
        event_analysis = state["query_results"].get("event_analysis", "")
        
        response = await llm.ainvoke(template.format(
            question=question,
            analysis=analysis,
            event_analysis=event_analysis
        ))
        state["final_answer"] = response.content
        return state
    
    return generate_final_answer


def create_graph(
    knowledge_graph_inst,
    entities_vdb,
    relationships_vdb,
    text_chunks_db,
    query_param,
    global_config: dict,
):
    # llm = ChatOpenAI(
    #     model=global_config.get("model_name", "gpt-4o"),
    #     temperature=global_config.get("temperature", 0.1)
    # )
    # llm = ChatOpenAI(
    #     model="Qwen3-32B",  # 模型名称与你测试成功的名称一致
    #     openai_api_base="http://10.10.202.242:2099/v1",
    #     openai_api_key="EMPTY",  # 使用环境变量
    #     temperature=0.2,
    #     max_tokens=2048
    # )
    llm = QwenNoThinkModel(
        model="Qwen3-32B",  # 模型名称与你测试成功的名称一致
        openai_api_base="http://10.10.202.242:2099/v1",
        openai_api_key="EMPTY",  # 使用环境变量
        temperature=0.05,
        max_tokens=16384,
    )

    tool_executor = create_tool_executor(
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param
    )

    workflow = StateGraph(MainAgentState)

    workflow.add_node("generate_keywords", create_keyword_generator(llm))
    # workflow.add_node("plan", create_query_planner(llm))
    workflow.add_node("query", execute_queries_wrapper(tool_executor))
    workflow.add_node("analyze", create_result_analyzer(llm))
    workflow.add_node("generate_answer", create_final_answer_generator(llm))
    workflow.add_node("reflect", create_reflection_analyzer(llm))

    # Event analysis
    workflow.add_node("event_query", create_event_query_executor(tool_executor))
    workflow.add_node("event_summarize", create_event_summarizer(llm))
    workflow.add_node("aggregate_events", create_event_aggregator(llm))

    # Add event reflection node
    workflow.add_node("event_reflect", create_event_reflection_analyzer(llm))

    # setup edges
    workflow.add_edge(START, "generate_keywords")
    workflow.add_edge("generate_keywords", "query")
    # workflow.add_edge("plan", "execute")
    workflow.add_edge("query", "analyze")
    workflow.add_edge("analyze", "reflect")

    workflow.add_conditional_edges(
        "reflect",
        lambda x: "query" if x["needs_more_info"] and x["iteration_count"] < MAX_ITERATIONS else "generate_answer",
        {
            "query": "query",
            "generate_answer": "generate_answer"
        }
    )

    # Event analysis
    workflow.add_edge("generate_keywords", "event_query")
    workflow.add_edge("event_query", "event_summarize")
    workflow.add_edge("event_summarize", "aggregate_events")
    workflow.add_edge("aggregate_events", "event_reflect")
    # Add conditional edges from event reflection
    workflow.add_conditional_edges(
        "event_reflect",
        lambda x: x["next"],
        {
            "event_query": "event_query",
            "generate_answer": "generate_answer"
        }
    )

    workflow.add_edge("generate_answer", END)

    workflow.set_entry_point("generate_keywords")

    from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
    graph = workflow.compile()
    # graph.llm = llm
    # ================ 自定义 Mermaid CLI 渲染函数 ======================
    def render_with_mmdc(mermaid_code: str, output_path: str):
        """使用 mmdc 将 Mermaid 代码渲染为 PNG"""
        with tempfile.NamedTemporaryFile(suffix=".mmd", mode="w", delete=False) as tmp:
            tmp.write(mermaid_code)
            tmp_path = tmp.name

        try:
            subprocess.run([
                "mmdc",
                "-i", tmp_path,
                "-o", output_path,
                "-t", "default",
                "-b", "white",
                "--quiet"  # 减少命令行输出
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Mermaid CLI 渲染失败: {e}")
        except FileNotFoundError:
            print("未找到 mmdc 命令，请确保已安装 Mermaid CLI: npm install -g @mermaid-js/mermaid-cli")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    try:
        # 获取 Mermaid 代码
        mermaid_code = graph.get_graph().draw_mermaid()
        # 使用本地 mmdc 渲染
        render_with_mmdc(mermaid_code, "agent_graph.png")
    except Exception as e:
        print(f"图表生成失败，但不影响主要功能: {e}")
    # =================================================================

    return graph

async def run_graph(
    query: str,
    knowledge_graph_inst,
    entities_vdb,
    relationships_vdb,
    text_chunks_db,
    query_param,
    global_config: dict,
) -> str:
    graph = create_graph(
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config
    )
    
    # prepare initial state
    initial_state = MainAgentState(
        messages=[HumanMessage(content=query)],
        next="plan",
        query_results={},
        final_answer="",
        iteration_count=0,
        needs_more_info=True,
        next_keywords=[]
    )
    # print("Graph nodes:", graph.nodes)  # 检查节点是否完整
    # print("Graph edges:", graph.get_graph().edges)  # 检查边是否正确连接

    # 临时测试 LLM 是否响应
    # print(graph.llm)
    # test_response = await graph.llm.ainvoke([HumanMessage(content="请说'pong'")])
    # print("LLM 测试响应:", test_response.content)
    # run the graph
    result = await graph.ainvoke(initial_state, debug=True)
    return result["final_answer"]


async def agent_query(query: str, knowledge_graph_inst, entities_vdb, relationships_vdb, text_chunks_db, query_param, global_config: dict) -> str:
    return await run_graph(query, knowledge_graph_inst, entities_vdb, relationships_vdb, text_chunks_db, query_param, global_config)