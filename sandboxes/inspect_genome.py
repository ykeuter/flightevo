import pickle
from pathlib import Path


FN = Path("logs/winner/member-4-winner.pickle")

with open(FN, "rb") as f:
    o = pickle.load(f)
print("hier")
