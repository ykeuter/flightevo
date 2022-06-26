import pickle


fn_in = "logs/bigrun-3/run-4/run-4-checkpoint-19-6.pickle"

with open(fn_in, "rb") as f:
    g = pickle.load(f)

print("finish")
