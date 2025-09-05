#!/usr/bin/env python3
# -*- encoding utf-8 -*-

"""
langextract抽取
"""
import json
import os
import time

# 修复可视化问题的方案
from utils import langextract as lx
from utils.knowLM_extract.langextract._base import LangextractConfig


class LangExtractor:
    def __init__(self):
        self.max_retries = 3
        self.project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")

    def splicing_prompt_format(self, prompt, prompt_format):
        return prompt + """
# 输出格式要求
以JSON格式输出，但请不要以```json````方式输出
请严格按照如下JSON字符串的格式回答：
""" + prompt_format

    def extract_list_of_dict(
            self, raw_prompt: str,
            result_format: dict,
            examples: list,
            input_text: str,
            langextract_config: LangextractConfig
    ):
        """
        从文本中提取知识，包含重试机制

        Args:
           raw_prompt (str): 提示词
           result_format (dict): 输出结果格式
           examples (list): 示例数据
           input_text (str): 输入文本
           langextract_config (LangextractConfig): 模型配置
        Returns:
           list(dict): 提取结果

        Raises:
           Exception: 如果所有重试都失败则抛出异常
        """
        # 检查输入文本是否为空
        if not input_text or not input_text.strip():
            print("警告: 输入文本为空或只包含空白字符")
            return []
            
        prompt = self.splicing_prompt_format(raw_prompt, json.dumps(result_format))
        # 初始化重试参数
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                print(f"尝试第 {attempt + 1}/{self.max_retries} 次提取...")

                # 使用附加模型language_model_type=CustomAPIModel时，需要为language_model_params添加参数"api_url"
                if langextract_config.language_model_type == lx.inference.CustomAPIModel:
                    langextract_config.config["api_url"] = langextract_config.api_url

                result = lx.extract(
                    text_or_documents=input_text,
                    prompt_description=prompt,
                    examples=examples,
                    model_id=langextract_config.model_name,
                    api_key=langextract_config.api_key,
                    language_model_type=langextract_config.language_model_type,
                    format_type=langextract_config.format_type,
                    max_char_buffer=langextract_config.max_char_buffer,
                    temperature=langextract_config.temperature,
                    fence_output=langextract_config.fence_output,
                    use_schema_constraints=langextract_config.use_schema_constraints,
                    batch_length=langextract_config.batch_length,
                    max_workers=langextract_config.max_workers,
                    additional_context=langextract_config.additional_context,
                    resolver_params=langextract_config.resolver_params,
                    debug=langextract_config.debug,
                    model_url=langextract_config.api_url,
                    extraction_passes=langextract_config.extraction_passes,
                    language_model_params=langextract_config.config
                )

                print(f"第 {attempt + 1} 次尝试成功!")
                return self.convert_annotated_document_to_dict(result)

            except Exception as e:
                last_exception = e
                print(f"第 {attempt + 1} 次尝试失败: {e}")

                # 如果不是最后一次尝试，等待一段时间再重试
                if attempt < self.max_retries - 1:
                    # 指数退避策略: 等待 2^attempt 秒
                    wait_time = 30 * (2 ** attempt)
                    print(f"等待 {wait_time} 秒后进行下一次尝试...")
                    time.sleep(wait_time)

                    # 特殊处理API限流错误
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        # 对于限流错误，等待更长时间
                        additional_wait = 5 * (attempt + 1)
                        print(f"检测到限流错误，额外等待 {additional_wait} 秒...")
                        time.sleep(additional_wait)

        # 所有重试都失败
        print(f"所有 {self.max_retries} 次尝试都失败了。")
        raise Exception(f"知识提取失败，已重试 {self.max_retries} 次。最后一次错误: {last_exception}") \
            from last_exception

    def convert_annotated_document_to_dict(
            self,
            annotated_doc: lx.data.AnnotatedDocument):
        """
        将 AnnotatedDocument 对象转换为标准字典结构。

        参数:
            annotated_doc: AnnotatedDocument 对象

        返回:
            dict: 包含文档原文和所有提取信息的字典
        """
        # 处理空输入
        if not annotated_doc:
            return {'text': '', 'extractions': []}

        # 初始化提取信息列表
        extractions_list = []

        # 遍历每个 Extraction 对象
        if annotated_doc.extractions:
            for extraction in annotated_doc.extractions:
                if not extraction:
                    continue
                    
                # 处理 char_interval
                char_interval_dict = None
                if hasattr(extraction, 'char_interval') and extraction.char_interval:
                    char_interval_dict = {
                        'start_pos': getattr(extraction.char_interval, 'start_pos', None),
                        'end_pos': getattr(extraction.char_interval, 'end_pos', None)
                    }

                # 处理 alignment_status
                alignment_status_value = None
                if hasattr(extraction, 'alignment_status') and extraction.alignment_status:
                    alignment_status_value = extraction.alignment_status.value \
                        if hasattr(extraction.alignment_status, 'value') else str(extraction.alignment_status)

                # 处理 token_interval
                token_interval_dict = None
                if hasattr(extraction, 'token_interval') and extraction.token_interval:
                    token_interval_dict = {
                        'start_index': getattr(extraction.token_interval, 'start_index', None),
                        'end_index': getattr(extraction.token_interval, 'end_index', None)
                    }

                # 构建每个提取项的字典
                extraction_dict = {
                    'extraction_class': getattr(extraction, 'extraction_class', ''),
                    'extraction_text': getattr(extraction, 'extraction_text', ''),
                    'char_interval': char_interval_dict,
                    'alignment_status': alignment_status_value,
                    'extraction_index': getattr(extraction, 'extraction_index', None),
                    'group_index': getattr(extraction, 'group_index', None),
                    'description': getattr(extraction, 'description', None),
                    'attributes': getattr(extraction, 'attributes', {}),  # attributes 本身应该是一个字典
                    'token_interval': token_interval_dict
                }
                extractions_list.append(extraction_dict)

        # 构建最终返回的文档字典
        doc_dict = {
            'text': getattr(annotated_doc, 'text', ''),
            'extractions': extractions_list
        }

        return doc_dict

# langExtractor = LangExtractor()
