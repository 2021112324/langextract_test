#!/usr/bin/env python3
# -*- encoding utf-8 -*-

"""
调用langextractor抽取图
"""
import os

from app.test.temp_text import text2
from utils.knowLM_extract.langextract._base import LangextractConfig
from utils.knowLM_extract.langextract.v2_langextractor import LangExtractor
from utils import langextract as lx
from utils.knowLM_extract.prompt.v2_format import node_format, edge_format
from utils.knowLM_extract.prompt.v2_prompt import prompt_for_node, prompt_for_edge
from models.v2_LLMs import ModelConfig
from utils.langextract.data import Extraction


class LangextractToGraph:
    def __init__(self, model_config: ModelConfig):
        self.langextract_config = LangextractConfig(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            api_url=model_config.api_url,
            config=model_config.config,
            max_char_buffer=500,
            temperature=0.3,
            batch_length=10,
            max_workers=5,
            debug=True,
        )
        self.langextractor = LangExtractor()

    # 抽取图谱
    async def extract_graph(
            self,
            prompt: str,
            schema: dict,
            examples: list,
            input_text: str
    ) -> dict | None:
        """
        参数结构示例
schema = {
    "nodes": [
        {
            "实体": "",
            "属性": ["属性1","属性2",...]
        },
        ...
    ],
    "edge": [
        {
            "关系": "",
            "主体": "",
            "谓词": "",
            "客体": ""
        },
        ...
    ]
}
examples = [{
    "text": "",
    "nodes": [
        {
            "extraction_class": "",
            "extraction_text": "",
            "attributes": {
                "属性1": "",
                "属性2": "",
                ...
            }
        },
        ...
    ],
    "edges": [
        {
            "extraction_class": "关系",
            "extraction_text": "",
            "attributes": {
                "主体": "调查准备",
                "谓词": "相关文件",
                "客体": "案件调查准备工作流程.docx"
            },
            ...
        },
        ...
    ]
},
...
]
        """
        node_schema = schema.get("nodes")
        edge_schema = schema.get("edges")
        node_examples = []
        edge_examples = []
        for example in examples:
            temp_node_example = {
                "text": example.get("text"),
                "extractions": example.get("nodes")
            }
            temp_edge_example = {
                "text": example.get("text"),
                "extractions": example.get("edges")
            }
            node_examples.append(temp_node_example)
            edge_examples.append(temp_edge_example)
        # 提取节点信息
        node_result = await self.extract_nodes(prompt, node_schema, node_examples, input_text)
        if node_result is None:
            return None
        print(type(node_result))
        print("节点信息:", node_result)
        extract_nodes = self.get_node_dict(node_result)
        print("节点加工信息：", extract_nodes)
        # 提取边信息
        edge_result = await self.extract_edges(prompt, extract_nodes, edge_schema, edge_examples, input_text)
        if edge_result is None:
            return None
        extract_edges = self.get_edge_dict(edge_result)
        extract_result = {
            "entities": extract_nodes,
            "relations": extract_edges
        }

        return extract_result

    # 抽取节点
    async def extract_nodes(
            self,
            prompt: str,
            node_schema: dict,
            examples: list,
            input_text: str
    ) -> list | None:
        """
        提取节点

        Args:
            prompt (str): 用户提示词
            node_schema (dict): 节点的本体schema
            examples (list): 节点的提取示例
            input_text (str): 提取文本

        Returns:
            list: extract_result 提取结果，失败时返回None
        """
        # 输入验证
        if not isinstance(prompt, str):
            print(f"Error: prompt should be a string, got {type(prompt)}")
            return None
            
        if not isinstance(node_schema, dict):
            print(f"Error: node_schema should be a dict, got {type(node_schema)}")
            return None
            
        if not isinstance(examples, list):
            print(f"Error: examples should be a list, got {type(examples)}")
            return None
            
        if not isinstance(input_text, str):
            print(f"Error: input_text should be a string, got {type(input_text)}")
            return None
            
        if not input_text.strip():
            print("Warning: input_text is empty or contains only whitespace")
            return []

        try:
            input_prompt = prompt_for_node(prompt, node_schema)
            input_examples = self.generate_examples(examples)
            extract_result = self.langextractor.extract_list_of_dict(
                input_prompt,
                node_format,
                input_examples,
                input_text,
                self.langextract_config
            )
            return extract_result
        except Exception as e:
            print(f"Error extracting nodes: {e}")
            return None

    # 抽取边
    async def extract_edges(
            self,
            prompt: str,
            nodes: list,
            edge_schema: dict,
            examples: list,
            input_text: str
    ) -> list | None:
        """
        提取边

        Args:
            prompt (str): 用户提示词
            nodes (list): 已提取的节点列表
            edge_schema (dict): 边的本体schema
            examples (list): 边的提取示例
            input_text (str): 提取文本

        Returns:
            list: extract_result 提取结果，失败时返回None
        """
        # 输入验证
        if not isinstance(prompt, str):
            print(f"Error: prompt should be a string, got {type(prompt)}")
            return None
            
        if not isinstance(nodes, list):
            print(f"Error: nodes should be a list, got {type(nodes)}")
            return None
            
        if not isinstance(edge_schema, dict):
            print(f"Error: edge_schema should be a dict, got {type(edge_schema)}")
            return None
            
        if not isinstance(examples, list):
            print(f"Error: examples should be a list, got {type(examples)}")
            return None
            
        if not isinstance(input_text, str):
            print(f"Error: input_text should be a string, got {type(input_text)}")
            return None
            
        if not input_text.strip():
            print("Warning: input_text is empty or contains only whitespace")
            return []

        try:
            # 遍历节点，记录节点列表
            node_list = []
            for i, node in enumerate(nodes):
                # 确保node是字典格式且包含name字段
                if isinstance(node, dict) and "name" in node:
                    node_list.append(node["name"])
                else:
                    print(f"Warning: node at index {i} should be a dict with 'name' key, got {type(node)}")

            input_prompt = prompt_for_edge(prompt, node_list, edge_schema)
            input_examples = self.generate_examples(examples)
            extract_result = self.langextractor.extract_list_of_dict(
                input_prompt,
                edge_format,
                input_examples,
                input_text,
                self.langextract_config
            )
            return extract_result
        except Exception as e:
            print(f"Error extracting edges: {e}")
            return None

    # 转换ExampleData实例
    def generate_examples(self, examples_list: list):
        """
        根据示例列表生成 ExampleData 对象列表

        Args:
            examples_list (list): 包含示例数据的列表

        Returns:
            list: ExampleData 对象列表
        """
        # 检查输入是否为列表
        if not isinstance(examples_list, list):
            print(f"Warning: examples_list should be a list, got {type(examples_list)}")
            return []

        examples = []
        for i, example in enumerate(examples_list):
            # 确保example是字典格式
            if not isinstance(example, dict):
                print(f"Warning: example at index {i} should be a dict, got {type(example)}")
                continue

            # 获取文本内容，默认为空字符串
            text = example.get("text", "")
            
            # 获取extractions字段
            extractions_data = example.get("extractions", [])
            if not isinstance(extractions_data, list):
                print(f"Warning: extractions at index {i} should be a list, got {type(extractions_data)}")
                extractions_data = []

            extractions = []
            for j, extraction in enumerate(extractions_data):
                # 确保extraction是字典格式
                if not isinstance(extraction, dict):
                    print(f"Warning: extraction at index {i}-{j} should be a dict, got {type(extraction)}")
                    continue

                # 获取必要字段，提供默认值
                extraction_class = extraction.get("extraction_class", "")
                extraction_text = extraction.get("extraction_text", "")
                attributes = extraction.get("attributes", {})
                
                # 确保attributes是字典格式
                if not isinstance(attributes, dict):
                    print(f"Warning: attributes at index {i}-{j} should be a dict, got {type(attributes)}")
                    attributes = {}

                # 跳过空的提取项
                if not extraction_class and not extraction_text:
                    print(f"Warning: skipping empty extraction at index {i}-{j}")
                    continue

                extractions.append(
                    lx.data.Extraction(
                        extraction_class=extraction_class,
                        extraction_text=extraction_text,
                        attributes=attributes
                    )
                )
            
            examples.append(
                lx.data.ExampleData(
                    text=text,
                    extractions=extractions
                )
            )
        return examples

    # 获取内的节点信息
    def get_node_dict(self, extraction_result: dict) -> list:
        """
        将LangExtract提取结果转换为节点列表格式

        Args:
            extraction_result (dict): LangExtract的提取结果

        Returns:
            list: 包含实体的列表，每个实体以指定格式表示
        """
        entities = []
        entity_ids_set = set()  # 用于去重

        # 确保输入是字典格式且包含extractions字段
        if not isinstance(extraction_result, dict):
            print(f"Warning: extraction_result should be a dict, got {type(extraction_result)}")
            return entities

        extractions = extraction_result.get("extractions", [])
        if not isinstance(extractions, list):
            print(f"Warning: extractions should be a list, got {type(extractions)}")
            return entities

        for extraction in extractions:
            # 确保extraction是字典格式
            if not isinstance(extraction, dict):
                print(f"Warning: extraction should be a dict, got {type(extraction)}")
                continue

            # 获取必要字段
            extraction_text = extraction.get('extraction_text', '')
            extraction_class = extraction.get('extraction_class', '')
            
            # 跳过空的提取文本或类名
            if not extraction_text or not extraction_class:
                continue

            # 获取属性，默认为空字典
            attributes = extraction.get('attributes', {})
            if not isinstance(attributes, dict):
                attributes = {}

            # 生成唯一ID
            entity_id = self.generate_unique_entity_id(extraction_text)
            
            # 避免重复实体
            if entity_id not in entity_ids_set:
                entity = {
                    "id": entity_id,
                    "name": extraction_text,
                    "label": extraction_class,
                    "properties": attributes
                }
                entities.append(entity)
                entity_ids_set.add(entity_id)

        return entities

    def get_edge_dict(self, extraction_result: dict) -> list:
        """
        将LangExtract提取结果转换为边列表格式

        Args:
            extraction_result (dict): LangExtract的提取结果

        Returns:
            list: 包含关系的列表，每个关系以指定格式表示
        """
        edges = []

        # 确保输入是字典格式且包含extractions字段
        if not isinstance(extraction_result, dict):
            print(f"Warning: extraction_result should be a dict, got {type(extraction_result)}")
            return edges

        extractions = extraction_result.get("extractions", [])
        if not isinstance(extractions, list):
            print(f"Warning: extractions should be a list, got {type(extractions)}")
            return edges

        for extraction in extractions:
            # 确保extraction是字典格式
            if not isinstance(extraction, dict):
                print(f"Warning: extraction should be a dict, got {type(extraction)}")
                continue

            # 检查是否为关系类型
            extraction_class = extraction.get('extraction_class', '')
            if extraction_class != "关系":
                continue

            # 获取属性，默认为空字典
            relation = extraction.get('attributes', {})
            if not isinstance(relation, dict):
                relation = {}

            # 检查必需的字段是否存在
            subject = relation.get("主体") or relation.get("主语")
            predicate = relation.get("谓词") or relation.get("谓语")
            obj = relation.get("客体") or relation.get("宾语")

            # 跳过缺少必要字段的关系
            if not subject or not predicate or not obj:
                continue

            edge = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "label": extraction_class,
            }
            edges.append(edge)

        return edges

    # 为实体生成唯一ID
    def generate_unique_entity_id(self, existing_ids: str) -> str:
        """
        为实体生成唯一ID

        Args:


        Returns:
            str: 唯一实体ID
        """
        # TODO: 添加生成唯一ID的逻辑
        unique_id = existing_ids
        return unique_id


if __name__ == "__main__":
    examples = []
    example = Extraction(
        extraction_class='关系',
        extraction_text='',
        char_interval=None,
        alignment_status=None,
        extraction_index=37,
        group_index=36,
        description=None,
        attributes={
            '主体': '检察提前介入监察：定位、功能与制度优化_金浩波 (1).pdf',
            '谓词': '相关业务',
            '客体': '建立与司法机关等在办理违纪案件和职务违法、职务犯罪案件中协作配合工作机制'
        }
    )
    model_config = ModelConfig(
        model_name="qwen-long",
        api_key="sk-742c7c766efd4426bd60a269259aafaf",
        api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        config={
            "temperature": 0.3,
            "fence_output": False,
            "use_schema_constraints": True,
            "debug": True,  # 启用调试模式
            "extraction_passes": 1,  # 通过多次传递提高召回率
            "max_workers": 5,  # 并行处理以提高速度
            "max_char_buffer": 500  # 较小的上下文以提高准确性
        }
    )
    test = LangextractToGraph(model_config)
    one = {
        'text': 'AnnotatedDocument(extractions=[',
    }
    examples.append(example)
    result = test.get_edge_dict(examples)
    print(result)
