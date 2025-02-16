from typing import Annotated, Any, Dict, List, Sequence, TypedDict
import operator
from langchain_core.messages import AIMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

MAX_ITERATIONS = 1

def merge_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries and concatenate string values if they have the same key"""
    result = dict1.copy()
    for k, v in dict2.items():
        if k in result and isinstance(v, str) and isinstance(result[k], str):
            # prevent duplicate values
            if v not in result[k]:
                result[k] = f"{result[k]}\n{v}"
        else:
            result[k] = v
    return result

def reduce_any(a:any, b:any) -> any:
    return b

def merge_messages(a: Sequence[Any], b: Sequence[Any]) -> Sequence[Any]:
    """merge messages and remove duplicates"""
    seen = set()
    result = []
    for msg in list(a) + list(b):
        msg_content = msg.content
        if msg_content not in seen:
            seen.add(msg_content)
            result.append(msg)
    return result

def reduce_iteration_count(a: int, b: int) -> int:
    """select the maximum iteration count"""
    return max(a, b)

def add_and_deduplicate(a: List[str], b: List[str]) -> List[str]:
    """add two lists and remove duplicates"""
    if "CLEAR" in b or "CLEAR" in a:
        return []
    return list(set(a + b))


# define the state schema
class BaseAgentState(TypedDict):
    # public state attributes
    messages: Annotated[Sequence[HumanMessage | AIMessage], merge_messages] # store the conversation history
    
class MainAgentState(BaseAgentState):
    # common_branch state attributes
    query_results: Annotated[Dict[str, Any], merge_dict]
    final_answer: Annotated[str, reduce_any]
    iteration_count: Annotated[int, reduce_iteration_count]
    needs_more_info: Annotated[bool, operator.or_]   
    next_keywords: Annotated[List[str], add_and_deduplicate]
    next: Annotated[str, reduce_any]
    
class EventAgentState(BaseAgentState):
    # event_branch state attributes
    event_search_results: Annotated[List[Any], operator.add]
    event_summaries: Annotated[List[str], operator.add]
    event_analysis: Annotated[str, operator.add]
    needs_more_events: Annotated[bool, operator.or_]
    missing_keywords: Annotated[List[str], operator.add]
    iteration_count: Annotated[int, reduce_iteration_count]
    next: Annotated[str, reduce_any]
    
# retry decorator
def create_retry_decorator():
    return retry(
        stop=stop_after_attempt(30),
        wait=wait_exponential(multiplier=1, min=4, max=300),
        reraise=True
    )