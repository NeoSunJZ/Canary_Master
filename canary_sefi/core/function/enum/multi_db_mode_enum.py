from enum import Enum


class MultiDatabaseMode(Enum):
    # 没有分库
    SIMPLE = "SIMPLE"

    # 每个攻击的结果数据单独存在于独立的数据库中
    EACH_ATTACK_ISOLATE_DB = "EACH_ATTACK_ISOLATE_DB"
