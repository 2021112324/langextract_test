#!/usr/bin/env python3
# -*- encoding utf-8 -*-

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    model_name: str = Field(..., description="模型名")
    api_key: str = Field(..., description="模型密钥")
    api_url: str = Field(..., description="模型地址")
    config: dict = Field(default_factory=dict, description="模型配置")
