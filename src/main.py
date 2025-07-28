from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# import iML1515 BiGG model
from cobra import io
model = io.load_json_model("iML1515.json")

solution = model.optimize()
print("Solution: ", solution)
print("Objective value: ", solution.objective_value)