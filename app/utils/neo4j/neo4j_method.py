#!/usr/bin/env python3
# -*- encoding utf-8 -*-
import os

from neo4j import GraphDatabase

neo4j_uri = os.getenv("NEO4J_URI", "bolt://60.205.171.106:7687")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "hit-wE8sR9wQ3pG1")
neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")  # 默认使用neo4j数据库


class Neo4j_method:
    def __init__(self, neo4j_uri, neo4j_username, neo4j_password, neo4j_database):
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database

    def save_kg_to_neo4j(
            self,
            kg_data: dict,
            graph_tag: str,
            filename: str = None,
            graph_level: str = "DocumentLevel"
    ):
        """
        将知识图谱数据保存到Neo4j数据库中，使用标签进行数据隔离

        Args:
            kg_data: 知识图谱数据，包含entities和relations
            graph_tag: 图谱标签
            filename: 文件名.txt，用于文档级分类
            graph_level: 存储层级 (DocumentLevel, DomainLevel, GlobalLevel)
        """
        # 创建数据库驱动
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password))

        try:
            with driver.session(database=self.neo4j_database) as session:
                # 开始事务
                with session.begin_transaction() as tx:
                    # 先创建所有实体节点，添加graph_tag属性和标签
                    for entity in kg_data.get('entities', []):
                        # 构建Cypher查询语句，使用MERGE确保相同ID的实体不会重复创建
                        query = (
                            f"MERGE (n:{graph_tag} {{id: $id}}) "
                            "SET n.name = $name, n.label = $label, n.graph_tag = $graph_tag"
                        )
                        
                        # 处理filename属性 - 以去重方式向列表增加新值
                        if filename:
                            query += ", n.filename = CASE WHEN n.filename IS NOT NULL THEN (CASE WHEN $filename IN n.filename THEN n.filename ELSE n.filename + $filename END) ELSE [$filename] END"
                            
                        # 处理graph_level属性 - 以去重方式向列表增加新值
                        query += ", n.graph_level = CASE WHEN n.graph_level IS NOT NULL THEN (CASE WHEN $graph_level IN n.graph_level THEN n.graph_level ELSE n.graph_level + $graph_level END) ELSE [$graph_level] END"

                        # 添加或更新其他属性 - 相同属性名取新值
                        properties = entity.get('properties', {})
                        for prop_key, prop_value in properties.items():
                            query += f", n.`{prop_key}` = ${prop_key}"

                        # 准备参数
                        params = {
                            'id': entity['id'],
                            'name': entity['name'],
                            'label': entity['label'],
                            'graph_tag': graph_tag,
                            'graph_level': graph_level,
                            **properties
                        }
                        
                        # 添加文件名参数（如果提供）
                        if filename:
                            params['filename'] = filename

                        tx.run(query, params)

                    # 创建关系
                    for relation in kg_data.get('relations', []):
                        subject_id = relation['subject']
                        predicate = relation['predicate']
                        object_id = relation['object']

                        # 获取关系的label属性（如果存在）
                        relation_label = relation.get('label', '')

                        # 处理关系名，确保符合Cypher命名规范
                        safe_predicate = ''.join(c if c.isalnum() else '_' for c in predicate)

                        # 使用MERGE确保相同节点间的关系不会重复创建
                        query = (
                            f"MATCH (a:{graph_tag} {{id: $subject_id}}), (b:{graph_tag} {{id: $object_id}}) "
                            f"MERGE (a)-[r:{safe_predicate}]->(b) "
                        )
                        
                        # 添加属性设置
                        if filename:
                            # 文件名以去重方式向列表增加新值
                            query += "SET r.graph_tag = $graph_tag, r.label = $relation_label, r.graph_level = CASE WHEN r.graph_level IS NOT NULL THEN (CASE WHEN $graph_level IN r.graph_level THEN r.graph_level ELSE r.graph_level + $graph_level END) ELSE [$graph_level] END, r.filename = CASE WHEN r.filename IS NOT NULL THEN (CASE WHEN $filename IN r.filename THEN r.filename ELSE r.filename + $filename END) ELSE [$filename] END "
                        else:
                            query += "SET r.graph_tag = $graph_tag, r.label = $relation_label, r.graph_level = CASE WHEN r.graph_level IS NOT NULL THEN (CASE WHEN $graph_level IN r.graph_level THEN r.graph_level ELSE r.graph_level + $graph_level END) ELSE [$graph_level] END "

                        query += "RETURN r"

                        params = {
                            'subject_id': subject_id, 
                            'object_id': object_id, 
                            'graph_tag': graph_tag,
                            'relation_label': relation_label,
                            'graph_level': graph_level
                        }
                        
                        # 添加文件名参数（如果提供）
                        if filename:
                            params['filename'] = filename

                        tx.run(query, **params)

                    # 提交事务
                    tx.commit()

                print(f"知识图谱数据已成功保存到数据库 {self.neo4j_database}，使用标签 {graph_tag}")

        finally:
            driver.close()

    def delete_kg_from_neo4j(self, graph_tag: str):
        """
        删除当前标签下的所有数据
        """
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password))

        try:
            with driver.session(database=self.neo4j_database) as session:
                # 删除当前标签下的所有数据
                session.run(f"MATCH (n:{graph_tag}) DETACH DELETE n")
                print(f"标签 {graph_tag} 下的所有数据已删除")
                return True

        except Exception as e:
            print(f"删除数据时出错: {e}")
            return False
        finally:
            driver.close()

    def clear_all_kg_data(self):
        """
        清空整个数据库中的所有知识图谱数据（谨慎使用）
        """
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password))

        try:
            with driver.session(database=self.neo4j_database) as session:
                # 删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                print(f"数据库 {self.neo4j_database} 中的所有数据已清空")
                return True

        except Exception as e:
            print(f"清空数据库时出错: {e}")
            return False
        finally:
            driver.close()


neo4j_method = Neo4j_method(neo4j_uri, neo4j_username, neo4j_password, neo4j_database)
