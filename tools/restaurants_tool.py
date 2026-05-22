"""
tools/restaurants_tool.py
-------------------------
LangChain tool for querying the Bangladeshi restaurants SQLite database.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), )
from tools.base_db_tool import BaseDBTool

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class RestaurantsDBTool(BaseDBTool):
    name: str = "RestaurantsDBTool"
    description: str = (
        "Use this tool to answer questions about Bangladeshi restaurants, "
        "food outlets, cafes, and eateries. "
        "Handles queries like: 'Top-rated restaurants in Chittagong?', "
        "'How many restaurants serve Chinese cuisine?', "
        "'List cheap restaurants in Dhaka with rating above 4', "
        "'Find vegetarian restaurants in Sylhet', "
        "'What cuisines are available in Cox's Bazar?'."
    )
    db_path: str = os.path.join(DATA_DIR, "restaurants.db")
    table_name: str = "restaurants"
    table_description: str = (
        "Contains records of Bangladeshi restaurants including name, cuisine type, "
        "location/district, area/neighbourhood, rating, and price range."
    )
