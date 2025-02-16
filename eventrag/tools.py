from typing import Optional, Any
from langchain.tools import BaseTool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .operate import _get_node_data, _get_edge_data, _build_query_context, _get_event_node_data


class KnowledgeGraphQueryInput(BaseModel):
    query: str = Field(..., description="The query text to search in the knowledge graph")
    top_k: int = Field(default=5, description="Number of top results to return")

class KnowledgeGraphTool(BaseTool):
    name: str = "knowledge_graph_query"
    description: str = """
    Query the knowledge graph to get related entities, relations, and text units.
    Useful when you need to find information from a structured knowledge base.
    Returns information in CSV format with entities, relations, and relevant text.
    """
    
    knowledge_graph_inst: Any = Field(description="Knowledge graph instance")
    entities_vdb: Any = Field(description="Vector database for entities")
    text_chunks_db: Any = Field(description="Database for text chunks")
    query_param: Any = Field(description="Query parameters")
    
    def __init__(
        self,
        knowledge_graph_inst,
        entities_vdb,
        text_chunks_db,
        query_param
    ):
        super().__init__(
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entities_vdb,
            text_chunks_db=text_chunks_db,
            query_param=query_param
        )

    async def _arun(
        self, query: str, top_k: Optional[int] = 5
    ) -> str:
        # Override top_k in query_param if provided
        self.query_param.top_k = top_k

        entities, relations, text_units = await _get_node_data(
            query,
            self.knowledge_graph_inst,
            self.entities_vdb,
            self.text_chunks_db,
            self.query_param
        )

        return f"""
Knowledge Graph Query Results:

=== Entities ===
{entities}

=== Relations ===
{relations}

=== Text Units ===
{text_units}
"""

    def _run(self, query: str, top_k: Optional[int] = 5) -> str:
        """Synchronous version - raises NotImplementedError as this tool is async-only"""
        raise NotImplementedError("KnowledgeGraphTool only supports async operations")
    
    
from typing import Optional, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class KnowledgeGraphEdgeQueryInput(BaseModel):
    keywords: str = Field(..., description="Keywords to search for relationships in the knowledge graph")
    top_k: int = Field(default=5, description="Number of top results to return")

class KnowledgeGraphEdgeTool(BaseTool):
    name: str = "knowledge_graph_edge_query"
    description: str = """
    Query the knowledge graph by relationship keywords to get related edges, entities, and text units.
    Useful when you need to find specific relationships or connections between entities in the knowledge base.
    Returns information in CSV format about relationships, connected entities, and relevant text.
    """
    
    knowledge_graph_inst: Any = Field(description="Knowledge graph instance")
    relationships_vdb: Any = Field(description="Vector database for relationships")
    text_chunks_db: Any = Field(description="Database for text chunks")
    query_param: Any = Field(description="Query parameters")
    
    def __init__(
        self,
        knowledge_graph_inst,
        relationships_vdb,
        text_chunks_db,
        query_param
    ):
        super().__init__(
            knowledge_graph_inst=knowledge_graph_inst,
            relationships_vdb=relationships_vdb,
            text_chunks_db=text_chunks_db,
            query_param=query_param
        )

    async def _arun(
        self, keywords: str, top_k: Optional[int] = 5
    ) -> str:
        # Override top_k in query_param if provided
        self.query_param.top_k = top_k

        entities, relations, text_units = await _get_edge_data(
            keywords,
            self.knowledge_graph_inst,
            self.relationships_vdb,
            self.text_chunks_db,
            self.query_param
        )

        return f"""
Knowledge Graph Edge Query Results:

=== Relationships ===
{relations}

=== Connected Entities ===
{entities}

=== Related Text Units ===
{text_units}
"""

    def _run(self, keywords: str, top_k: Optional[int] = 5) -> str:
        """Synchronous version - raises NotImplementedError as this tool is async-only"""
        raise NotImplementedError("KnowledgeGraphEdgeTool only supports async operations")
    
"""
# Initialize the tool
kg_edge_tool = KnowledgeGraphEdgeTool(
    knowledge_graph_inst=your_graph_instance,
    relationships_vdb=your_vector_db,
    text_chunks_db=your_text_chunks_db,
    query_param=your_query_param
)

# Use in async context
result = await kg_edge_tool.arun(keywords="collaboration between", top_k=5)

# Both tools can be used together in a Langchain agent to provide comprehensive knowledge graph querying capabilities, with the node-based tool focusing on entity queries and this edge-based tool focusing on relationship queries.
"""

class KeywordsQueryInput(BaseModel):
    keywords: str = Field(..., description="Keywords list to search for in the knowledge graph")

class KeywordsQueryTool(BaseTool):
    name: str = "keywords_query"
    description: str = "Query the knowledge graph by keywords to get related entities and text units."
    
    knowledge_graph_inst: Any = Field(description="Knowledge graph instance")
    entities_vdb: Any = Field(description="Vector database for entities")
    relationships_vdb: Any = Field(description="Vector database for relationships")
    text_chunks_db: Any = Field(description="Database for text chunks")
    query_param: Any = Field(description="Query parameters")
    
    def __init__(self, knowledge_graph_inst, entities_vdb, relationships_vdb, text_chunks_db, query_param):
        super().__init__(
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            text_chunks_db=text_chunks_db,
            query_param=query_param
        )
    
    async def _arun(self, keywords: list[str]) -> str:
        # Override top_k in query_param if provided
        result = await _build_query_context(
            keywords,
            self.knowledge_graph_inst,
            self.entities_vdb,
            self.relationships_vdb,
            self.text_chunks_db,
            self.query_param
        )
        return result
    
    def _run(self, keywords: str, top_k: Optional[int] = 5) -> str:
        """Synchronous version - raises NotImplementedError as this tool is async-only"""
        raise NotImplementedError("KnowledgeGraphEdgeTool only supports async operations")

class EventGraphQueryInput(BaseModel):
    query: str = Field(..., description="The query text to search for events in the knowledge graph")
    top_k: int = Field(default=5, description="Number of top results to return")

class EventGraphTool(BaseTool):
    name: str = "event_graph_query"
    description: str = """
    Query the knowledge graph to find event-related entities, their relationships, and associated text units.
    Useful when you need to find information about specific events or actions and their connections.
    Returns information in CSV format about events, their relationships, and relevant text.
    """
    
    knowledge_graph_inst: Any = Field(description="Knowledge graph instance")
    entities_vdb: Any = Field(description="Vector database for entities")
    relationships_vdb: Any = Field(description="Vector database for relationships")
    text_chunks_db: Any = Field(description="Database for text chunks")
    query_param: Any = Field(description="Query parameters")
    
    def __init__(
        self,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param
    ):
        super().__init__(
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            text_chunks_db=text_chunks_db,
            query_param=query_param
        )

    async def _arun(
        self, query: str, top_k: Optional[int] = 5
    ) -> list[str]:
        # Override top_k in query_param if provided
        self.query_param.top_k = top_k * 2

        subgraphs = await _get_event_node_data(
            query,
            self.knowledge_graph_inst,
            self.entities_vdb,
            self.relationships_vdb,
            self.text_chunks_db,
            self.query_param
        )

        return subgraphs

    def _run(self, query: str, top_k: Optional[int] = 5) -> str:
        """Synchronous version - raises NotImplementedError as this tool is async-only"""
        raise NotImplementedError("EventGraphTool only supports async operations")
