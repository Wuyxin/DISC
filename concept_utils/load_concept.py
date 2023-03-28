import pickle
from concept_utils import ConceptBank

def load_concept(concept_path, device):
    concept_bank = ConceptBank(pickle.load(open(concept_path, "rb")), device=device)
    concepts = concept_bank.concept_info.concept_names
    print("Number of concepts: ", len(concepts))
    return concept_bank