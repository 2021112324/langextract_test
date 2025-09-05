#!/usr/bin/env python3
# -*- encoding utf-8 -*-
import json

from utils.knowLM_extract.prompt.v2_defination import entityDefinition, relationDefinition


def prompt_for_node(raw_prompt: str, node_schema: dict):
    prompt = raw_prompt + """
**以下内容最重要**
# 提取内容
为了提高抽取效率，我们将抽取任务分为两部分
本次对话仅提取实体和其内部的属性
本体任务提取的实体schema如下：
# 实体schema
""" + json.dumps(node_schema) + """
# 定义
实体的定义如下
""" + entityDefinition + """
# 注意事项
1. 严格按照定义的实体类型进行提取
2. 只提取文本中明确表达的信息，不要进行推断
3. 保持文本原始表述，不要改写
# 输出要求：
1. 严格按以下JSON格式输出，不要添加任何额外文本或解释
2. 确保JSON语法正确，可以被直接解析
3. 所有字符串使用双引号(")而非单引号(')
4. 不要包含任何Markdown格式或代码块标记
"""
    return prompt

def prompt_for_edge(raw_prompt: str, node_list: list, schema: dict):
    prompt = raw_prompt + """
**以下内容最重要**
# 提取内容
为了提高抽取效率，我们将抽取任务分为两部分
本次对话请根据实体提取关系
本体任务已提取的实体如下：
""" + str(node_list) + """
本体任务提取的关系schema如下：
# 关系schema
""" + json.dumps(schema) + """
# 定义
实体的定义如下
""" + relationDefinition + """
# 注意事项
1. 严格按照定义的关系类型进行提取
2. 只提取文本中明确表达的信息，或根据文本明确内容进行推断
3. 保持文本原始表述，不要改写
# 输出要求：
1. 严格按以下JSON格式输出，不要添加任何额外文本或解释
2. 确保JSON语法正确，可以被直接解析
3. 所有字符串使用双引号(")而非单引号(')
4. 不要包含任何Markdown格式或代码块标记
"""
    return prompt
