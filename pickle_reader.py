import pickle
import pprint

f = open("best_genome.pkl", "rb")
best_genome = pickle.load(f)
pprint.pprint(best_genome)
