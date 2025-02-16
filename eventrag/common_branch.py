from langgraph.prebuilt import ToolInvocation
import logging
import json
import re
from .prompt import PROMPTS
from .graph_state import MAX_ITERATIONS, MainAgentState, create_retry_decorator
logger = logging.getLogger(__name__)
  

def create_query_planner(llm):
    template = """Based on the user's question and the extracted keywords, decide how to use the following tools to query:

1. knowledge_graph_query: query entity-related information
2. knowledge_graph_edge_query: query relationship-related information

User question: {question}
Extracted keywords: {keywords}

Please decide which tool is more appropriate or whether you need to use both tools at the same time.
Return the decision in JSON format: {{"tools": ["tool_name"], "queries": ["specific query terms"]}}
"""
    
    @create_retry_decorator()
    async def query_planner(state: MainAgentState) -> MainAgentState:
        messages = state["messages"]
        question = messages[-1].content
        keywords = state["query_results"]["keywords"]
        
        response = await llm.ainvoke(template.format(
            question=question,
            keywords=json.dumps(keywords, ensure_ascii=False)
        ))
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            plan = json.loads(match.group(0))
            state["query_results"]["plan"] = plan
        else:
            logger.error("No JSON-like structure found in the result.")
            return PROMPTS["fail_response"]
        return state
    
    return query_planner

# execute queries
def execute_queries_wrapper(tool_executor):
    @create_retry_decorator()
    async def execute_queries(state: MainAgentState) -> MainAgentState:
        logger.info(f"Current state: {state}")
        # plan = state["query_results"]["plan"]
        results = []

        keywords_to_query = state["next_keywords"] if state["next_keywords"] else state["query_results"]["keywords"].split(" ,")
        
        # for tool_name, query in zip(plan["tools"], plan["queries"]):
        #     tool_invocation = ToolInvocation(
        #         tool=tool_name,
        #         tool_input=query
        #     )
        #     result = await tool_executor.ainvoke(tool_invocation)
        #     results.append(result)

        keywords_query_invocation = ToolInvocation(
            tool="keywords_query",
            tool_input=str(keywords_to_query)
        )
        keywords_query = await tool_executor.ainvoke(keywords_query_invocation)
        results.append(keywords_query)
        
        if "raw_results" in state["query_results"]:
            state["query_results"]["raw_results"] += results
        else:
            state["query_results"]["raw_results"] = results
        state["query_results"]["keywords"] = f"{state['query_results']['keywords']} ,{', '.join(state['next_keywords'])}"
        state["next_keywords"] = ["CLEAR"]  # clear the keywords for the next iteration
        return state
    
    return execute_queries

# result analysis
def create_result_analyzer(llm):
    template = """Analyze the information retrieved from the knowledge graph and perform reasoning:

    Previous analysis (if any):
    {previous_analysis}

    New query results:
    {results}

    Original question:
    {question}

    Please summarize all findings and perform comprehensive reasoning analysis. Consider both previous and new information.
    """
    
    @create_retry_decorator()
    async def analyze_results(state: MainAgentState) -> MainAgentState:
        results = state["query_results"]["raw_results"]
        question = state["messages"][-1].content
        # get previous analysis (if any)
        previous_analysis = state["query_results"].get("analysis", "")
        
        response = await llm.ainvoke(template.format(
            previous_analysis=previous_analysis,
            results=results,
            question=question
        ))
        state["query_results"]["analysis"] = response.content
        return state
    
    return analyze_results

def create_reflection_analyzer(llm):
    template = """Analyze the currently collected information and determine if it is sufficient to answer the user's question:

    Original question: {question}
    Current analysis results: {analysis}
    Searched keywords: {searched_keywords}
    Current iteration count: {iteration_count}

    Please consider:
    1. Is the existing information sufficient to answer the question?
    2. What key information is still missing?
    3. What new keywords need to be queried to supplement the information?

    Please return in JSON format, the json should contain the following fields:
     - needs_more_info: a boolean value indicating whether more information is needed
     - missing_info: a list of missing information
     - new_keywords: a list of new keywords to be queried based on the missing information, no more than 5
    """
    
    @create_retry_decorator()
    async def reflect_on_results(state: MainAgentState) -> MainAgentState:
        if state["iteration_count"] >= MAX_ITERATIONS:
            state["needs_more_info"] = False
            state["next_keywords"] = ["CLEAR"]
            return state
            
        question = state["messages"][-1].content
        analysis = state["query_results"].get("analysis", "")
        searched_keywords = state["query_results"].get("keywords", "")
        
        response = await llm.ainvoke(template.format(
            question=question,
            analysis=analysis,
            searched_keywords=searched_keywords,
            iteration_count=state["iteration_count"]
        ))
        
        try:
            reflection = json.loads(re.search(r"\{.*\}", response.content, re.DOTALL).group(0))
            state["needs_more_info"] = reflection["needs_more_info"]
            if state["needs_more_info"]:
                state["next_keywords"] = reflection["new_keywords"]
                state["query_results"]["keywords"] = f"{searched_keywords} ,{', '.join(state['next_keywords'])}"
            else:
                state["next_keywords"] = ["CLEAR"]
            state["iteration_count"] += 1
        except:
            state["needs_more_info"] = False
            state["next_keywords"] = ["CLEAR"]
            
        return state
    
    return reflect_on_results

def workflow_router(state: MainAgentState) -> MainAgentState:
    # if the information is sufficient or the maximum number of iterations is reached, go to the answer generation stage
    if not state["needs_more_info"] or int(state["iteration_count"]) >= MAX_ITERATIONS:
        state["next"] = "generate_answer"
    else:
        state["next"] = "plan"  # go back to the planning stage
    return state

