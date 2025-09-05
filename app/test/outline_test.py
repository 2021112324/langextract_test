
import re
import os

from models.v2_LLMs import ModelConfig
from utils.knowLM_extract.langextract.v2_langextrct_to_graph import LangextractToGraph
from utils.knowLM_extract.prompt.v2_format import edge_format
from utils.neo4j.neo4j_method import neo4j_method


class GraphService:
    def parse_outline_to_graph(self, file_path) -> dict:
        """
        解析大纲文件并构建实体关系图

        Args:
            file_path (str): 大纲文件路径

        Returns:
            dict: 包含实体和关系的图结构
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        entities = []
        relations = []
        entity_ids = {}  # 用于跟踪已创建的实体ID

        # 用于跟踪层级结构
        level_stack = []

        # 创建根节点（层级0）
        root_entity = {
            'id': '案件监督管理类业务',
            'name': '案件监督管理类业务',
            'label': '业务节点',
            'properties': {
                '层级': '0'
            }
        }
        entities.append(root_entity)
        entity_ids[root_entity['id']] = root_entity
        level_stack.append(root_entity)

        # 解析大纲内容
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-'):
                # 跳过标题行、注释行和文件列表行
                continue

            # 提取层级和内容
            level = 0
            content = line

            # 计算缩进层级
            if re.match(r'^\d+\.', line):
                level = 1
                content = re.sub(r'^\d+\.\s*', '', line)
            elif re.match(r'^[（(]\d+[）)]', line):
                level = 2
                content = re.sub(r'^[（(]\d+[）)]\s*', '', line)
            elif re.match(r'^[a-zA-Z]\.', line):
                level = 3
                content = re.sub(r'^[a-zA-Z]\.\s*', '', line)
            elif re.match(r'^[-－]', line):
                level = 4
                content = re.sub(r'^[-－]\s*', '', line)

            if not content:
                continue

            # 处理层级变化
            while len(level_stack) > level:
                if level_stack:
                    level_stack.pop()

            # 创建实体
            entity_id = content
            if entity_id not in entity_ids:
                entity = {
                    'id': entity_id,
                    'name': content,
                    'label': '业务节点',
                    'properties': {
                        '层级': str(level)
                    }
                }
                entities.append(entity)
                entity_ids[entity_id] = entity

            # 建立关系
            if level_stack:
                parent_entity = level_stack[-1]
                relation = {
                    'subject': parent_entity['name'],
                    'predicate': '包含',
                    'object': content
                }
                relations.append(relation)

            # 将当前实体加入层级栈
            # 确保层级栈的长度与当前层级一致
            if len(level_stack) > level:
                level_stack = level_stack[:level]
            level_stack.append(entity_ids[entity_id])

        return {
            'entities': entities,
            'relations': relations
        }

    def get_leaf_nodes(self, graph_dict):
        """
        获取图中的叶子节点（没有子节点的节点）

        Args:
            graph_dict (dict): 图结构字典，包含entities和relations

        Returns:
            list: 叶子节点列表
        """
        # 获取所有实体名称
        all_entities = set()
        for entity in graph_dict['entities']:
            all_entities.add(entity['name'])

        # 获取所有作为父节点的实体名称（有"包含"关系的源节点）
        parent_entities = set()
        for relation in graph_dict['relations']:
            if relation['predicate'] == '包含':
                parent_entities.add(relation['subject'])

        # 叶子节点是没有作为任何"包含"关系源节点的实体
        leaf_nodes = []
        for entity in graph_dict['entities']:
            if entity['name'] not in parent_entities:
                leaf_nodes.append(entity)

        return leaf_nodes

    def parse_filenames_to_entities(self, file_path):
        """
        解析文件名列表文件，将每行文件名处理成实体并保存到dict数据中

        Args:
            file_path (str): 文件名列表文件路径

        Returns:
            dict: 包含实体的字典
        """
        entities = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue

            # 移除行首的 "- " 或 "-" 符号
            if line.startswith('- '):
                filename = line[2:].strip()
            elif line.startswith('-'):
                filename = line[1:].strip()
            else:
                filename = line

            # 跳过空文件名
            if not filename:
                continue

            # 获取文件扩展名
            file_parts = filename.split('.')
            if len(file_parts) > 1:
                file_format = file_parts[-1].lower()
            else:
                file_format = 'unknown'

            # 创建实体
            entity = {
                'id': filename,
                'name': filename,
                'label': '文件',
                'properties': {
                    '文件格式': file_format
                }
            }
            entities.append(entity)

        return {
            'entities': entities
        }

def main():
    """
    主函数，用于运行大纲解析功能
    """
    # 指定要解析的大纲文件路径
    file_path = "案件监督管理类业务大纲与文件列表.txt"
    filename_list_path = "文件名.txt"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，请确认文件路径")
        return
    if not os.path.exists(filename_list_path):
        print(f"文件 {filename_list_path} 不存在，请确认文件路径")
        return

    # 创建GraphService实例
    graph_service = GraphService()
    model_config = ModelConfig(
        model_name="qwen-long",
        api_key="sk-742c7c766efd4426bd60a269259aafaf",
        api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    # 创建测试实例
    extractor = LangextractToGraph(model_config)
    extractor.langextract_config.temperature = 0.5
    extractor.langextract_config.max_char_buffer = 10000
    extractor.langextract_config.batch_length = 5
    extractor.langextract_config.max_workers = 3

    # 调用解析函数
    result = graph_service.parse_outline_to_graph(file_path)
    # 获取叶子节点
    leaf_nodes = graph_service.get_leaf_nodes(result)
    # print(leaf_nodes)
    # 叶子节点业务列表
    leaf_node_list = []
    for leaf_node in leaf_nodes:
        leaf_node_list.append(leaf_node['name'])
    # 获取文件名
    filename_entities = graph_service.parse_filenames_to_entities(filename_list_path)
    # print(filename_entities)
    # 文件名列表
    filename_list = []
    for filename_entity in filename_entities['entities']:
        filename_list.append(filename_entity['name'])

    print("叶子节点业务列表：", leaf_node_list)
    print("文件名列表：", filename_list)

    input_prompt = """
# 角色
您是一个案件监督管理业务专家，能够通过案件监督管理类业务大纲对相关文件进行分类和业务关联，用于构建案件监督管理业务分类知识图谱。

# 任务说明
输入的内容包括三个部分：业务大纲、叶子结点业务列表、文件名列表
请根据大纲内的业务节点，为材料文件进行业务关联和分类，包括如下任务：
1. 加载业务大纲节点层级关系，理解对文件分类的业务大纲结构和含义，为叶子结点业务的理解做铺垫
2. 结合业务大纲，遍历叶子结点业务列表中的每个节点业务，理解其含义和分类范围
3. 遍历文件名列表中的每个文件名，对每个文件都进行如下操作：分析其业务领域，根据文件名，猜测该文件可归类到哪一业务节点，将其与叶子结点业务列表中的业务节点进行关联、分类，构建出"文件名 - 相关业务 - 业务节点"关系

# **重要要求**
1. 请尽最大可能得将每个文件名都与叶子结点业务列表中的业务节点关联
2. 文件名关联的业务节点必须是是叶子结点业务列表中的业务节点,即"文件名 - 相关业务 - 业务节点"中的"业务节点"必须是"叶子结点业务列表"中的"业务节点"！！！

本体任务提取的关系schema如下：
# 关系schema
[
    {
        "关系": "相关业务",
        "主体": "文件名",
        "谓词": "相关文件",
        "客体": "业务节点"
    },
]
# 输出要求：
1. 严格按以下JSON格式输出，不要添加任何额外文本或解释
2. 确保JSON语法正确，可以被直接解析
3. 所有字符串使用双引号(")而非单引号(')
4. 不要包含任何Markdown格式或代码块标记
"""
    text = """
# 案件监督管理类业务
1. 线索管理
（1） 本机关收到的问题线索
（2） 同级党委（党组）、上级机关交办的问题线索
（3） 本机关主要负责人批示的问题线索
（4） 巡视巡察机构、党委政法委、审计机关、执法机关、司法机关等单位会商后移交的问题线索
（5） 其他需要集中管理的问题线索
2. 组织协调
（1） 内部查办案件流程协调
    a. 反腐败协调小组会议组织筹办
    b. 联系成员单位相关职能部门
    c. 参加调研，重要文稿、文件起草
    d. 指导下级反腐败领导小组开展工作
    e. 反腐败协调小组领导同志交办的其他事项
（2） 外部各机关协作配合组织协调
    a. 建立与其他机关建立健全线索移交等机制
    b. 建立执法机关、金融机构等协助开展监督监察、审查调查工作机制
    c. 建立与司法机关等在办理违纪案件和职务违法、职务犯罪案件中协作配合工作机制
    d. 与军队有关部门开展协作配合工作机制
    e. 其他需要建立健全的协作配合机制
3. 督促办理
（1） 加强督办
（2） 督促办结
（3） 督促改正
4. 统计分析
（1） 重点领域违纪违法问题和案件
（2） 重点岗位违纪违法问题和案件
5. 监督检查
（1） 问题线索
    a. 抽查问题线索管理台账
    b. 汇总问题线索、处置情况
（2） 安全监管
    a. 留置场所安全监管
        - 人员安排
        - 审查调查工作情况
        - 服务保障
        - 场所和周边安全
    b. “走读式”谈话安全管控
        - 审批程序
        - 制定安全预案
        - 遵守谈话时限要求
        - 全程管控谈话安全等
（3） 措施使用监督
    a. 措施使用条件
    b. 审批权限
    c. 办理程序
    d. 文书手续等
（4） 涉案财物监管
    a. 抽查涉案财物查扣手续
    b. 逐案核对涉案财物处置情况
    c. 核查录音录像等
（5） 查询平台监督
    a. 平台建设
    b. 平台管理
    c. 平台使用情况
"""

    examples = [
        {
            "text": "# 案件监督管理类业务\n1. 线索管理\n（1） 本机关收到的问题线索\n2. 组织协调\n（1） 内部查办案件流程协调\na. 反腐败协调小组会议组织筹办\n# 叶子结点业务列表\n["
                    "'本机关收到的问题线索','反腐败协调小组会议组织筹办']\n# 文件名列表\n['获取问题线索后流程指导.txt','反腐败协调小组会议筹办事项.docx']",
            "extractions": [
                {
                    "extraction_class": "关系",
                    "extraction_text": "",
                    "attributes": {
                        "主体": "获取问题线索后流程指导.txt",
                        "谓词": "相关业务",
                        "客体": "本机关收到的问题线索"
                    },
                },
                {
                    "extraction_class": "关系",
                    "extraction_text": "",
                    "attributes": {
                        "主体": "反腐败协调小组会议筹办事项.docx",
                        "谓词": "相关业务",
                        "客体": "反腐败协调小组会议组织筹办"
                    },
                },
            ]
        },
    ]
    input_examples = extractor.generate_examples(examples)
    input_text = """
# 业务大纲
""" + text + """
# 叶子结点业务列表
""" + str(leaf_node_list) + """
# 文件名列表
""" + str(filename_list)

    relation_result = extractor.langextractor.extract_list_of_dict(
        input_prompt,
        edge_format,
        input_examples,
        input_text,
        extractor.langextract_config
    )
    print("\033[95m这部分应该是list[dict]\n\n" + str(relation_result) + "\033[0m")

    extract_edges = extractor.get_edge_dict(relation_result)
    print("\033[92m" + str(extract_edges) + "\033[0m")
    extract_relations = []
    for relation in extract_edges:
        temp_relation = {
            "subject": relation.get("object"),
            "predicate": "相关文件",
            "object": relation.get("subject"),
            "label": "相关文件"
        }
        extract_relations.append(temp_relation)
    # print(extract_relations)
    extract_result = {
        "entities": result["entities"] + filename_entities["entities"],
        "relations": result["relations"] + extract_relations
    }
    print(extract_result)

    # neo4j_method.save_kg_to_neo4j(extract_result, "outline_mix2", "案件监督管理类业务.txt")

    # 返回结果以供其他模块使用
    return result


if __name__ == "__main__":
    main()


