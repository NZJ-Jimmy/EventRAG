import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
import numpy as np
from typing import Union
from eventrag.utils import logger
from ..base import BaseVectorStorage

from pymilvus import MilvusClient
from tenacity import retry, stop_after_attempt, wait_exponential

@dataclass
class MilvusVectorDBStorge(BaseVectorStorage):
    @staticmethod
    def create_collection_if_not_exist(
        client: MilvusClient, collection_name: str, **kwargs
    ):
        if client.has_collection(collection_name):
            return
        client.create_collection(
            collection_name, max_length=64, id_type="string", **kwargs
        )

    def __post_init__(self):
        self._client = MilvusClient(
            uri=os.environ.get(
                "MILVUS_URI",
                os.path.join(self.global_config["working_dir"], "milvus_lite.db"),
            ),
            user=os.environ.get("MILVUS_USER", ""),
            password=os.environ.get("MILVUS_PASSWORD", ""),
            token=os.environ.get("MILVUS_TOKEN", ""),
            db_name=os.environ.get("MILVUS_DB_NAME", ""),
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        MilvusVectorDBStorge.create_collection_if_not_exist(
            self._client,
            self.namespace,
            dimension=self.embedding_func.embedding_dim,
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=300))
    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        

        list_data = []
        contents = []
        for k, v in data.items():
            if not v or "content" not in v:
                logger.warning(f"Skipping invalid data entry for key {k}")
                continue
            list_data.append({
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            })
            contents.append(v["content"])
        
        if not contents:
            logger.warning("No valid content to process")
            return []

        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            try:
                result = await self.embedding_func(batch)
                if result is None:
                    logger.error("Embedding function returned None")
                    return np.array([])
                pbar.update(1)
                return result
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                pbar.update(1)
                return np.array([])

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)
        
        valid_embeddings = []
        valid_data = []
        current_idx = 0
        
        for batch_idx, embeddings in enumerate(embeddings_list):
            if embeddings is not None and embeddings.size > 0: 
                batch_size = len(batches[batch_idx])
                valid_embeddings.append(embeddings)
                valid_data.extend(list_data[current_idx:current_idx + batch_size])
            current_idx += len(batches[batch_idx])
        
        if not valid_embeddings:
            logger.error("All embedding generations failed")
            return []
            
        embeddings = np.concatenate(valid_embeddings)
        
        for i, d in enumerate(valid_data):
            d["vector"] = embeddings[i]
        
        if not valid_data:
            logger.warning("No valid data to insert after filtering")
            return []
            
        results = self._client.upsert(collection_name=self.namespace, data=valid_data)
        logger.info(f"Successfully inserted {len(results)} vectors to {self.namespace}")
        return results

    async def query(self, query, top_k=5):
        embedding = await self.embedding_func([query])
        results = self._client.search(
            collection_name=self.namespace,
            data=embedding,
            limit=top_k,
            output_fields=list(self.meta_fields),
            search_params={"metric_type": "COSINE", "params": {"radius": 0.2}},
        )
        # print(results)
        return [
            {**dp["entity"], "id": dp["id"], "distance": dp["distance"]}
            for dp in results[0]
        ]

    async def find_similar_entity(self, entity_name: str, entity_description: str, similarity_threshold: float = 0.9) -> Union[str, None]:
        similar_entities = await self.query(entity_description, top_k=1)
        if similar_entities and similar_entities[0]["distance"] > similarity_threshold:
            logger.debug(f"✅Found similar entity: {similar_entities[0]['entity_name']} with similarity: {similar_entities[0]['distance']}")
            return similar_entities[0]["entity_name"]
        logger.debug(f"No similar entity found for {entity_name}")
        return None
