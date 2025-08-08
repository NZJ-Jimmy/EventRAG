from dataclasses import field
import os
import pathlib
import logging
from eventrag import EventRAG, QueryParam
from eventrag.llm import openai_complete_if_cache, openai_embedding, openai_complete, jina_embedding
from eventrag.utils import wrap_embedding_func_with_attrs
import numpy as np
from dotenv import load_dotenv
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="ch", # Specify French recognition model with the lang parameter
    use_doc_orientation_classify=False, # Disable document orientation classification model
    use_doc_unwarping=False, # Disable text image unwarping model
    use_textline_orientation=False, # Disable text line orientation classification model
)

import asyncio
# Load environment variables from .env file
load_dotenv(override=True)


def read_file_content(file_path):
    """
    读取各种类型文件的内容
    - txt: 直接读取文本
    - docx/doc: 使用文档处理库提取文本
    - jpg/png/jpeg/bmp/pdf: 使用PaddleOCR识别文本
    """
    file_extension = pathlib.Path(file_path).suffix.lower()

    # 文本文件处理
    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    # Word文档处理
    elif file_extension == '.docx':
        try:
            from docx import Document
            doc = Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except ImportError:
            logger.error("请安装python-docx库: pip install python-docx")
            raise

    elif file_extension == '.doc':
        try:
            import textract
            return textract.process(file_path).decode('utf-8')
        except ImportError:
            try:
                # 尝试使用pywin32（仅Windows系统）
                import win32com.client
                word = win32com.client.Dispatch("Word.Application")
                word.visible = False
                doc = word.Documents.Open(str(pathlib.Path(file_path).absolute()))
                text = doc.Content.Text
                doc.Close()
                word.Quit()
                return text
            except ImportError:
                logger.error("请安装textract或pywin32库处理doc文件")
                raise

    # 图像和PDF处理（使用PaddleOCR）
    elif file_extension in ['.jpg', '.png', '.jpeg', '.bmp', '.pdf']:
        result = ocr.predict(file_path)
        text = ""
        for res in result:
            text += "\n".join(res['rec_texts']) + "\n"
        # 提取OCR识别的文本
        return text

    else:
        raise ValueError(f"不支持的文件类型: {file_extension}")

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./chat_history_preprocess"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Qwen embedding configuration
@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=512)
async def jina_embedding(texts, **kwargs):
    """Qwen embedding wrapper using OpenAI-compatible interface"""
    return await openai_embedding(
        texts=texts,
            base_url="http://10.10.202.242:2097/v1", # automatically use the base URL from environment variables
        # api_key="sk-1",  # automatically use the API key from environment variables
        model="bce-embedding-base_v1",  # Qwen's text embedding model
        dimensions=768,  # Specify dimensions for Qwen embedding
    )

# overlap_token_size=self.chunk_overlap_token_size, 100
# max_token_size=self.chunk_token_size,1200
# tiktoken_model=self.tiktoken_model_name,
rag = EventRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_complete,  # Use OpenAI API, automatically use API key from environment variables
    embedding_func=jina_embedding,  # Use Qwen text embedding
    embedding_batch_num=10,
    llm_model_name="Qwen3-32B",  # Use qwen-plus model
    # llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
    # chunk_token_size=50,
    # chunk_overlap_token_size=10,
    node2vec_params= {
            "dimensions": 768,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
)


failed_ls = []
logger = logging.getLogger("eventrag-bioplanner")
# for file in pathlib.Path("./chat_history_split").glob("*.txt"):
#     try:
#         with open(file, "r", encoding="utf-8") as f:
#             rag.insert(f.read())
#         logger.info(f"🎇Inserted {file}")
#     except Exception as e:
#         failed_ls.append(file)
#         logger.error(f"💥Failed to insert {file}: {e}")
#
# logger.info(f"💥Failed to insert {len(failed_ls)} files, {failed_ls}")

# with open("test/test.txt", "r", encoding="utf-8") as f:
#     rag.insert(f.read())
for file in pathlib.Path(WORKING_DIR).glob("*.*"):
    rag.insert(read_file_content(file))

# 打印高rank节点信息
# print(rag.print_rank())
# exit()
# asyncio.run(rag.print_top_100_nodes_by_rank())
# Perform naive search
# print(
#     rag.query("李欣是谁？", param=QueryParam(mode="local"))
# )
# 李欣和张洁是什么关系

# import time
#
# # 记录开始时间
# start_time = time.time()
#
# # Perform multi-event reasoning
# print(
#     rag.query("把群里的人做一个画像（例如：谁在主导这件事情？每个人的分工？）", param=QueryParam(mode="agent"))
# )
# # 计算并打印执行时间
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"查询执行时间: {elapsed_time:.2f} 秒")