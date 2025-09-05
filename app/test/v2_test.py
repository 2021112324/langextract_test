import asyncio
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.test.temp_text import text2, text1
from utils.knowLM_extract.langextract.v2_langextrct_to_graph import LangextractToGraph
from models.v2_LLMs import ModelConfig


async def main():
    model_config = ModelConfig(
        model_name="qwen-long",
        api_key="sk-742c7c766efd4426bd60a269259aafaf",
        api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
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
        "edges": [
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
    result = await extractor.extract_graph(
        prompt=prompt,
        schema=schema,
        examples=examples,
        input_text=text2
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
