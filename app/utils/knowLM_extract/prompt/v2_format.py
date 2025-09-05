#!/usr/bin/env python3
# -*- encoding utf-8 -*-

node_format1 = """
{
    "extractions": [
        {
            "类别实体": "提取的实体",
            "类别实体_attributes": {
                "属性名1": "属性值1",
                "属性名2": "属性值2",
                ...
            }
        }
    ]
}
"""

edge_format1 = """
{
    "extractions": [
        {
            "关系": "关系文本",
            "关系_attributes": {
                "主体": "华为", 
                "谓词": "研发", 
                "客体": "麒麟芯片"
            }
        }
    ]
}
"""

node_format = node_format1
edge_format = edge_format1
