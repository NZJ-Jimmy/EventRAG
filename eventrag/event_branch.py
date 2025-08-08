from langgraph.prebuilt import ToolInvocation
from .graph_state import EventAgentState, MAX_ITERATIONS, create_retry_decorator
import json
import re
import logging

logger = logging.getLogger(__name__)

def create_event_query_executor(tool_executor):
    async def execute_event_query(state: EventAgentState) -> EventAgentState:
        keywords = state.get("missing_keywords", [])
        if not keywords:  # if there are no missing_keywords, use the last message as the keyword
            keywords = [state["messages"][-1].content]
            
        event_query_invocation = ToolInvocation(
            tool="event_graph_query",  # Using EventGraphTool
            tool_input=str(keywords)
        )
        results = await tool_executor.ainvoke(event_query_invocation)
        state["event_search_results"] = [results]  # update event_search_results
        state["missing_keywords"] = []
        return state
    return execute_event_query

def create_event_summarizer(llm):
    template = """Analyze the following event-related information in relation to the original question:

    Original question: {question}
    Event information: {event_info}

    Provide a concise summary focusing on the temporal and causal relationships between events. Keep the same language as `Original question`.
    """

    @create_retry_decorator()
    async def summarize_events(state: EventAgentState) -> EventAgentState:
        question = state["messages"][-1].content
        events = state["event_search_results"]  # use the search results as the event information
        
        response = await llm.ainvoke(template.format(
            question=question,
            event_info=events
        ))
        
        state["event_summaries"] = [response.content]  # update event_summaries
        return state
    
    return summarize_events

def create_event_aggregator(llm):
    template = """Synthesize the following event summaries into a coherent timeline or causal chain:

    Original question: {question}
    Event summaries:
    {summaries}

    Provide a comprehensive analysis that emphasizes:
    1. Temporal relationships between events
    2. Cause-and-effect relationships
    3. Key actors and their roles in the events
    
    Keep the same language as `Original question`.
    """

    @create_retry_decorator()
    async def aggregate_event_summaries(state: EventAgentState) -> EventAgentState:
        question = state["messages"][-1].content
        summaries = state.get("event_summaries", [])
        
        response = await llm.ainvoke(template.format(
            question=question,
            summaries="\n".join(summaries)
        ))
        
        state["event_analysis"] = response.content  # update event_analysis
        return state
    
    return aggregate_event_summaries

def create_event_reflection_analyzer(llm):
    template = """Analyze the current event-based reasoning and identify any gaps:

    Original question: {question}
    Current event analysis: {event_analysis}

    Please evaluate:
    1. Is event-based reasoning appropriate for this question?
    2. Are there any missing temporal or causal relationships?
    3. What additional event-related keywords should we search for? The number of event-related keywords should be less than 3.

    Respond in the following JSON format:
    {{
        "needs_more_events": boolean,
        "missing_keywords": ["keyword1", "keyword2"],
        "reasoning": "Your explanation here"
    }}
    """

    @create_retry_decorator()
    async def reflect_on_events(state: EventAgentState) -> EventAgentState:
        question = state["messages"][-1].content
        event_analysis = state.get("event_analysis", "")
        
        response = await llm.ainvoke(template.format(
            question=question,
            event_analysis=event_analysis
        ))
        
        try:
            if "{{" in response.content:
                response.content = response.content.replace("{{", "{").replace("}}", "}")
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                reflection = json.loads(match.group(0))
            else:
                logger.error("No JSON-like structure found in the response")
                reflection = {"needs_more_events": False, "missing_keywords": []}
            state["needs_more_events"] = reflection["needs_more_events"]
            state["missing_keywords"] = reflection["missing_keywords"]
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            
            if state["needs_more_events"] and state["iteration_count"] < MAX_ITERATIONS:
                state["next"] = "event_query"
            else:
                state["next"] = "generate_answer"
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reflection response: {e}")
            state["needs_more_events"] = False
            state["next"] = "generate_answer"
            
        return state
    
    return reflect_on_events
