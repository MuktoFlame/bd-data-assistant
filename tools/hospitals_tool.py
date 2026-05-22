"""
tools/hospitals_tool.py
-----------------------
LangChain tool for querying the Bangladeshi hospitals SQLite database.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.base_db_tool import BaseDBTool

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class HospitalsDBTool(BaseDBTool):
    name: str = "HospitalsDBTool"
    description: str = (
        "Use this tool to answer questions about Bangladeshi hospitals, "
        "medical centres, clinics, and healthcare facilities. "
        "Handles queries like: 'How many hospitals are in Dhaka?', "
        "'Which hospital has the most beds in Sylhet?', "
        "'List all government hospitals in Rajshahi', "
        "'Find hospitals with more than 200 beds', "
        "'What are the contact details for Chittagong Medical College Hospital?'."
    )
    db_path: str = os.path.join(DATA_DIR, "hospitals.db")
    table_name: str = "hospitals"
    table_description: str = (
        "Contains records of Bangladeshi hospitals including name, location/district, "
        "division, hospital type (government/private/specialised), bed count, "
        "and contact information."
    )
