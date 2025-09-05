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
        提取边

        Args:
            prompt (str): 用户提示词,
            node_schema (dict): 节点的本体schema,
            examples (list): 节点的提取示例,
            input_text (str): 提取文本

        Returns:
            list: extract_result 提取结果
        """
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
            print(f"Error: {e}")
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
            prompt (str): 用户提示词,
            nodes (list): 已提取的节点列表,
            edge_schema (dict): 边的本体schema,
            examples (list): 边的提取示例,
            input_text (str): 提取文本

        Returns:
            list: extract_result 提取结果
        """
        try:
            # 遍历节点，记录节点列表
            node_list = []
            for node in nodes:
                node_list.append(node["name"])
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
            print(f"Error: {e}")
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
        examples = []
        for example in examples_list:
            extractions = []
            for extraction in example["extractions"]:
                extractions.append(
                    lx.data.Extraction(
                        extraction_class=extraction["extraction_class"],
                        extraction_text=extraction["extraction_text"],
                        attributes=extraction["attributes"]
                    )
                )
            examples.append(
                lx.data.ExampleData(
                    text=example["text"],
                    extractions=extractions
                )
            )
        return examples

    # 获取内的节点信息
    def get_node_dict(self, extraction_result: list) -> list:
        """
        将LangExtract提取结果转换为节点列表格式

        Args:
            extraction_result (list): LangExtract的提取结果

        Returns:
            list: 包含实体的列表，每个实体以指定格式表示
        """
        entities = []
        entity_ids_set = set()  # 用于去重

        # 遍历提取结果
        for item in extraction_result:
            # 确保item是字典格式
            if not isinstance(item, dict):
                if hasattr(item, 'to_dict'):
                    item = item.to_dict()
                else:
                    continue  # 跳过无法处理的项目

            # 检查是否有extractions字段且不为空
            extractions = item.get("extractions", [])

            # 如果extractions为空但text字段包含AnnotatedDocument信息
            if not extractions and "AnnotatedDocument" in item.get("text", ""):
                # 从text字段中提取extractions信息
                text_content = item.get("text", "")
                # 查找Extraction对象
                import re
                # 使用正则表达式提取Extraction对象
                extraction_pattern = r"Extraction\(([^)]+)\)"
                matches = re.findall(extraction_pattern, text_content)

                # 解析每个Extraction对象
                for match in matches:
                    # 手动解析Extraction参数
                    extraction_data = {}
                    # 解析extraction_class
                    class_match = re.search(r"extraction_class='([^']+)'", match)
                    if class_match:
                        extraction_data["extraction_class"] = class_match.group(1)

                    # 解析extraction_text
                    text_match = re.search(r"extraction_text='([^']+)'", match)
                    if text_match:
                        extraction_data["extraction_text"] = text_match.group(1)

                    # 解析attributes
                    attr_match = re.search(r"attributes=(\{[^}]+\})", match)
                    if attr_match:
                        try:
                            # 安全地评估attributes字典
                            extraction_data["attributes"] = eval(attr_match.group(1))
                        except:
                            extraction_data["attributes"] = {}

                    # 将解析的数据添加到extractions列表
                    extractions.append(extraction_data)

            # 处理extractions中的每个Extraction对象
            for extraction in extractions:
                # 确保extraction不是None且有必要的属性
                if extraction is None:
                    continue

                # 获取实体基本信息
                entity_class = (getattr(extraction, 'extraction_class', '')
                                if hasattr(extraction, 'extraction_class')
                                else extraction.get('extraction_class', '') if isinstance(extraction, dict) else '')
                entity_text = (getattr(extraction, 'extraction_text', '')
                               if hasattr(extraction, 'extraction_text')
                               else extraction.get('extraction_text', '') if isinstance(extraction, dict) else '')

                # 获取属性
                attributes = (getattr(extraction, 'attributes', {})
                              if hasattr(extraction, 'attributes')
                              else extraction.get('attributes', {}) if isinstance(extraction, dict) else {})

                # 确保attributes是字典格式
                if not isinstance(attributes, dict):
                    attributes = {}

                # 只处理实体类型（排除关系类型）
                if entity_class != "关系" and entity_text and entity_class:
                    # 生成基于实体文本的唯一ID
                    entity_id = self.generate_unique_entity_id(entity_text)
                    # 避免重复实体
                    if entity_id not in entity_ids_set:
                        entity = {
                            "id": entity_id,
                            "name": entity_text,
                            "label": entity_class,
                            "properties": attributes,
                        }
                        entities.append(entity)
                        entity_ids_set.add(entity_id)

        return entities

    def get_edge_dict(self, extraction_result: list) -> list:
        """
        将LangExtract提取结果转换为边列表格式

        Args:
            extraction_result (list): LangExtract的提取结果

        Returns:
            list: 包含关系的列表，每个关系以指定格式表示
        """
        relations = []

        # 遍历提取结果
        for item in extraction_result:
            # 确保item是字典格式
            if not isinstance(item, dict):
                if hasattr(item, 'to_dict'):
                    item = item.to_dict()
                else:
                    continue  # 跳过无法处理的项目

            # 检查是否有extractions字段
            extractions = item.get("extractions", [])

            # 如果extractions为空但text字段包含AnnotatedDocument信息
            if not extractions and "AnnotatedDocument" in item.get("text", ""):
                # 从text字段中提取extractions信息
                text_content = item.get("text", "")
                # 查找Extraction对象
                import re
                # 使用正则表达式提取Extraction对象
                extraction_pattern = r"Extraction\(([^)]+)\)"
                matches = re.findall(extraction_pattern, text_content)

                # 解析每个Extraction对象
                for match in matches:
                    # 手动解析Extraction参数
                    extraction_data = {}
                    # 解析extraction_class
                    class_match = re.search(r"extraction_class='([^']+)'", match)
                    if class_match:
                        extraction_data["extraction_class"] = class_match.group(1)

                    # 解析extraction_text
                    text_match = re.search(r"extraction_text='([^']+)'", match)
                    if text_match:
                        extraction_data["extraction_text"] = text_match.group(1)

                    # 解析attributes
                    attr_match = re.search(r"attributes=(\{[^}]+\})", match)
                    if attr_match:
                        try:
                            # 安全地评估attributes字典
                            extraction_data["attributes"] = eval(attr_match.group(1))
                        except:
                            extraction_data["attributes"] = {}

                    # 将解析的数据添加到extractions列表
                    extractions.append(extraction_data)

            # 直接处理extractions中的每个Extraction对象
            for extraction in extractions:
                # 确保extraction不是None且有必要的属性
                if extraction is None:
                    continue

                # 获取实体基本信息
                entity_class = (getattr(extraction, 'extraction_class', '')
                                if hasattr(extraction, 'extraction_class')
                                else extraction.get('extraction_class', '') if isinstance(extraction, dict) else '')

                # 获取属性
                attributes = (getattr(extraction, 'attributes', {})
                              if hasattr(extraction, 'attributes')
                              else extraction.get('attributes', {}) if isinstance(extraction, dict) else {})

                # 确保attributes是字典格式
                if not isinstance(attributes, dict):
                    attributes = {}

                # 只处理关系类型
                if entity_class == "关系" or (
                        entity_class and entity_class != "关系" and attributes.get('主体') and attributes.get(
                        '谓词') and attributes.get('客体')):
                    # 从attributes中获取关系三元组
                    subject = attributes.get('主体') or attributes.get('主语')
                    predicate = attributes.get('谓词') or attributes.get('谓语')
                    object = attributes.get('客体') or attributes.get('宾语')

                    # 只添加有效的关系
                    if subject and predicate and object:
                        relation = {
                            "subject": subject,
                            "predicate": predicate,
                            "object": object,
                            "label": entity_class if entity_class else predicate,
                        }
                        relations.append(relation)

        return relations

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

    # 创建测试实例
    extractor = LangextractToGraph(model_config)

    prompt = """
# 角色
您是一个法律文本分析专家，能够从法律法规、案件描述等文本中准确提取实体、属性和关系信息，用于构建违纪违法行为知识图谱。

# 任务说明
从各类违纪违法行为相关文本中提取构建知识图谱所需的信息，包括案件、法律法规条款、违法行为、构成要件等实体及其相互关系。
"""

    schema = {
        "nodes": [
            {
                "实体": "案件",
                "属性": ["案件编号", "发生时间", "涉案人员", "处理结果", "案件类型"]
            },
            {
                "实体": "法律法规",
                "属性": ["法规名称", "条款编号", "颁布机构", "生效时间", "适用范围"]
            },
            {
                "实体": "违法行为",
                "属性": ["行为类型", "严重程度", "所属领域", "处罚方式"]
            },
            {
                "实体": "构成要件",
                "属性": ["要件类型", "具体内容", "证明标准"]
            },
            {
                "实体": "处罚措施",
                "属性": ["处罚类型", "处罚程度", "执行机关"]
            },
            {
                "实体": "主体",
                "属性": ["主体类型", "身份信息", "责任程度"]
            },
        ],
        "edge": [
            {
                "关系": "违反",
                "主体": "案件",
                "谓词": "违反",
                "客体": "法律法规"
            },
            {
                "关系": "禁止",
                "主体": "法律法规",
                "谓词": "禁止",
                "客体": "违法行为"
            },
            {
                "关系": "定义",
                "主体": "法律法规",
                "谓词": "定义",
                "客体": "违法行为"
            },
            {
                "关系": "包含",
                "主体": "违法行为",
                "谓词": "包含",
                "客体": "构成要件"
            },
            {
                "关系": "涉及",
                "主体": "案件",
                "谓词": "涉及",
                "客体": "违法行为"
            },
            {
                "关系": "构成",
                "主体": "案件",
                "谓词": "构成",
                "客体": "违法行为"
            },
            {
                "关系": "对应",
                "主体": "违法行为",
                "谓词": "对应",
                "客体": "处罚措施"
            },
            {
                "关系": "涉及",
                "主体": "主体",
                "谓词": "涉及",
                "客体": "案件"
            },
            {
                "关系": "实施",
                "主体": "主体",
                "谓词": "实施",
                "客体": "案件"
            },
        ]
    }
    examples = [
        {
            "text": "2023年5月，某科技公司因违反《网络安全法》第45条规定，非法收集用户个人信息，被网信部门处以50万元罚款。",
            "nodes": [
                {
                    "extraction_class": "案件",
                    "extraction_text": "某科技公司非法收集用户个人信息",
                    "attributes": {
                        "案件编号": "网信违处字[2023]第125号",
                        "发生时间": "2023年5月",
                        "涉案人员": "某科技公司",
                        "处理结果": "罚款50万元",
                        "案件类型": "网络安全违法"
                    }
                },
                {
                    "extraction_class": "法律法规",
                    "extraction_text": "《网络安全法》第45条",
                    "attributes": {
                        "法规名称": "网络安全法",
                        "条款编号": "第45条",
                        "颁布机构": "全国人民代表大会常务委员会",
                        "生效时间": "2017年6月1日",
                        "适用范围": "网络运营者收集、使用个人信息"
                    }
                }
            ],
            "edges": [
                {
                    "extraction_class": "关系",
                    "extraction_text": "某科技公司因违反《网络安全法》第45条规定，非法收集用户个人信息",
                    "attributes": {
                        "主体": "某科技公司非法收集用户个人信息",
                        "谓词": "违反",
                        "客体": "《网络安全法》第45条"
                    }
                }
            ],
        },
        {
            "text": "非法收集用户个人信息行为的构成要件包括：主体为网络运营者、客体为用户个人信息、行为方式为非法收集。",
            "nodes": [
                {
                    "extraction_class": "构成要件",
                    "extraction_text": "主体为网络运营者",
                    "attributes": {
                        "要件类型": "主体要件",
                        "具体内容": "行为人必须是网络运营者",
                        "证明标准": "提供网络服务的组织或个人"
                    }
                }
            ],
            "edges": [
                {
                    "extraction_class": "关系",
                    "extraction_text": "非法收集用户个人信息行为的构成要件包括：主体为网络运营者、客体为用户个人信息、行为方式为非法收集",
                    "attributes": {
                        "主体": "非法收集用户个人信息",
                        "谓词": "需要/包含",
                        "客体": "主体为网络运营者"
                    }
                }
            ]
        }
    ]

    # 执行测试
    result = extractor.extract_graph(
        prompt=prompt,
        schema=schema,
        examples=examples,
        input_text=text2
    )

    print(result)
