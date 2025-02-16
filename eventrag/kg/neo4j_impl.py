import asyncio
import os
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict
import inspect
from eventrag.utils import logger
from ..base import BaseGraphStorage, BaseVectorStorage
from neo4j import (
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
)


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class Neo4JStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with neo4j in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()
        URI = os.environ["NEO4J_URI"]
        USERNAME = os.environ["NEO4J_USERNAME"]
        PASSWORD = os.environ["NEO4J_PASSWORD"]
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            URI, auth=(USERNAME, PASSWORD)
        )
        return None

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            await self._driver.close()

    async def index_done_callback(self):
        print("KG successfully indexed.")

    async def has_node(self, node_id: str) -> bool:
        entity_name_label = node_id.strip('"')
        async with self._driver.session() as session:
            query = f"MATCH (n:`{entity_name_label}`) RETURN count(n) AS node_count"
            result = await session.run(query)
            records = await result.values()
            return records[0][0] > 0

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')
        async with self._driver.session() as session:
            query = (
                f"MATCH (a:`{entity_name_label_source}`)-[r]-(b:`{entity_name_label_target}`) "
                "RETURN COUNT(r) AS edge_count"
            )
            result = await session.run(query)
            records = await result.values()
            return records[0][0] > 0

    async def get_node(self, node_id: str) -> Union[dict, None]:
        async with self._driver.session() as session:
            entity_name_label = node_id.strip('"')
            query = f"MATCH (n:`{entity_name_label}`) RETURN n LIMIT 1"
            result = await session.run(query)
            record = await result.single()
            if record:
                node = record["n"]
                node_dict = dict(node)
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}: query: {query}, result: {node_dict}"
                )
                return node_dict
            return None

    async def node_degree(self, node_id: str) -> int:
        entity_name_label = node_id.strip('"')

        async with self._driver.session() as session:
            query = f"""
                MATCH (n:`{entity_name_label}`)
                RETURN COUNT{{ (n)--() }} AS totalEdgeCount
            """
            result = await session.run(query)
            record = await result.single()
            if record:
                edge_count = record["totalEdgeCount"]
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_count}"
                )
                return edge_count
            else:
                return None

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        entity_name_label_source = src_id.strip('"')
        entity_name_label_target = tgt_id.strip('"')
        src_degree = await self.node_degree(entity_name_label_source)
        trg_degree = await self.node_degree(entity_name_label_target)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            f"{inspect.currentframe().f_code.co_name}:query:src_Degree+trg_degree:result:{degrees}"
        )
        return degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')
        """
        Find all edges between nodes of two given labels

        Args:
            source_node_label (str): Label of the source nodes
            target_node_label (str): Label of the target nodes

        Returns:
            list: List of all relationships/edges found
        """
        async with self._driver.session() as session:
            query = f"""
            MATCH (start:`{entity_name_label_source}`)-[r]->(end:`{entity_name_label_target}`)
            RETURN properties(r) as edge_properties, type(r) as relationship_type
            LIMIT 1
            """.format(
                entity_name_label_source=entity_name_label_source,
                entity_name_label_target=entity_name_label_target,
            )

            result = await session.run(query)
            record = await result.single()
            if record:
                properties = dict(record["edge_properties"])
                properties["relationship_name"] = record["relationship_type"]
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{properties}"
                )
                return properties
            else:
                return None

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        node_label = source_node_id.strip('"')

        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: List of dictionaries containing edge information
        """
        query = f"""MATCH (n:`{node_label}`)
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, r, connected, type(r) as relationship_type"""
        async with self._driver.session() as session:
            results = await session.run(query)
            edges = []
            async for record in results:
                source_node = record["n"]
                connected_node = record["connected"]
                relationship_type = record["relationship_type"]

                source_label = (
                    list(source_node.labels)[0] if source_node.labels else None
                )
                target_label = (
                    list(connected_node.labels)[0]
                    if connected_node and connected_node.labels
                    else None
                )

                if source_label and target_label:
                    edges.append((source_label, target_label, relationship_type))

            return edges

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=300),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        label = node_id.strip('"')
        properties = node_data

        async def _do_upsert(tx: AsyncManagedTransaction):

            event_type = properties.get('event_type', properties.get('entity_type', properties.get('type', 'Node')))
            query = f"""
            MERGE (n:`{label}`:`{event_type}`)
            SET n += $properties
            """
            await tx.run(query, properties=properties)
            logger.debug(
                f"Upserted event node with label '{label}' and properties: {properties}"
            )

        try:
            async with self._driver.session() as session:
                await session.execute_write(_do_upsert)
        except Exception as e:
            logger.error(f"Error during event upsert: {str(e)}")
            # raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=300),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
            )
        ),
    )
    async def upsert_edge(
        self, source_event_id: str, target_event_id: str, edge_data: Dict[str, Any]
    ):

        source_label = source_event_id.strip('"')
        target_label = target_event_id.strip('"')
        edge_properties = edge_data
        
        relationship_type = edge_properties.get("relationship_name", "RELATED_TO").upper()
        if "relationship_name" in edge_properties:
            del edge_properties["relationship_name"]

        async def _do_upsert_edge(tx: AsyncManagedTransaction):
            query = f"""
            MATCH (source:`{source_label}`)
            WITH source
            MATCH (target:`{target_label}`)
            MERGE (source)-[r:`{relationship_type}`]->(target)
            SET r += $properties
            RETURN r
            """
            await tx.run(query, properties=edge_properties)
            logger.debug(
                f"Upserted edge from '{source_label}' to '{target_label}' with type '{relationship_type}' and properties: {edge_properties}"
            )

        try:
            async with self._driver.session() as session:
                await session.execute_write(_do_upsert_edge)
        except Exception as e:
            logger.error(f"Error during relationship upsert: {str(e)}")
            # raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")

    async def find_similar_entity(
        self, 
        entity_name: str, 
        entity_description: str, 
        entities_vdb: BaseVectorStorage,
        similarity_threshold: float = 0.9
    ) -> Union[str, None]:
        """
        Find existing similar entity based on description embedding similarity using vector database.
        
        Args:
            entity_name: Name of the new entity
            entity_description: Description of the new entity
            entities_vdb: Vector database instance for entity similarity search
            similarity_threshold: Threshold for considering entities similar
            
        Returns:
            The name of the most similar existing entity if similarity > threshold, else None
        """
        return await entities_vdb.find_similar_entity(
            entity_name, 
            entity_description, 
            similarity_threshold
        )
