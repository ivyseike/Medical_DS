import pickle
import numpy as np

goal_set = pickle.load(open("goal_set.p", "rb"))
train_set = goal_set["train"]
test_set = goal_set["test"]

# action_set = pickle.load(open("action_set.p", "rb"))
# print(action_set)

disease_symp = pickle.load(open("disease_symptom.p", "rb"))
slot_set = pickle.load(open("slot_set.p","rb"))
disease_set = pickle.load(open("disease_set.p", "rb"))

# with open("disease_set.p", "wb") as f:
#     pickle.dump(disease_set, f)
arr = np.zeros([len(disease_symp), len(slot_set)-1])
sym_disease = np.zeros([len(slot_set)-1, len(disease_symp)])
prior_sym = np.zeros([len(slot_set)-1])

# for key, val in disease_symp.items():
#     idx = val['index']
#     for sym, freq in val['symptom'].items():
#         sym_idx = slot_set[sym]
#         arr[idx][sym_idx] = freq

# print(arr)

# print(slot_set)
# print(len(slot_set))

for s in train_set:
    # print(s["consult_id"])
    # print(s["disease_tag"])
    # print(s["goal"])
    for k, v in s["goal"]["explicit_inform_slots"].items():
        if v is True:
            arr[disease_set[s["disease_tag"]]][slot_set[k]] += 1
            prior_sym[slot_set[k]] += 1
            sym_disease[slot_set[k]][disease_set[s["disease_tag"]]] += 1
    for k, v in s["goal"]["implicit_inform_slots"].items():
        if v is True:
            arr[disease_set[s["disease_tag"]]][slot_set[k]] += 1
            prior_sym[slot_set[k]] += 1
            sym_disease[slot_set[k]][disease_set[s["disease_tag"]]] += 1


sum_ = np.sum(arr, axis=1)
sum_2 = np.sum(sym_disease, axis=1)
sum_3 = np.sum(prior_sym, axis=0)

# print(sum_)
# print(sum_2)
# print(sum_3)
for i in range(len(disease_symp)):
    arr[i] /= sum_[i]

for i in range(len(slot_set)-1):
    prior_sym[i] /= sum_3
    sym_disease[i] /= sum_2[i]

print(arr)
print(sym_disease)
print(prior_sym)

np.savetxt("prior_symptom.txt", prior_sym, fmt="%.8f")
np.savetxt("symptom_given_disease.txt", arr, fmt="%.8f")
np.savetxt("disease_given_symptom.txt", sym_disease, fmt="%.8f")

