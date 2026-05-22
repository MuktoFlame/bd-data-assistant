"""
tools/institutions_tool.py
--------------------------
LangChain tool for querying the Bangladeshi institutions SQLite database.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.base_db_tool import BaseDBTool

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class InstitutionsDBTool(BaseDBTool):
    name: str = "InstitutionsDBTool"
    description: str = (
        "Use this tool to answer questions about Bangladeshi educational and "
        "governmental institutions such as universities, colleges, schools, and "
        "government bodies. "
        "Handles queries like: 'How many universities are in Dhaka?', "
        "'List government institutions in Chittagong', "
        "'Which division has the most colleges?', "
        "'Find public universities established before 2000'."
    )
    db_path: str = os.path.join(DATA_DIR, "institutions.db")
    table_name: str = "institutions"
    table_description: str = (
        "Contains records of Bangladeshi educational and governmental institutions "
        "including name, location/district, division, type (university/college/school/govt), "
        "and year established."
    )
