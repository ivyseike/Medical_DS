import pickle
import numpy as np

goal_set = pickle.load(open("goal_set.p", "rb"))
train_set = goal_set["train"]
test_set = goal_set["test"]

# slot_set = pickle.load(open("slot_set.p","rb"))
# # disease_set = pickle.load(open("disease_set.p", "rb"))

# print(slot_set)
# del slot_set['disease']

# symptoms = [None for i in range(len(slot_set))]

# for key, idx in slot_set.items():
#     symptoms[idx] = key

# print(symptoms)

# with open("symptoms.txt", "w") as f:
#  for i in range(len(symptoms)):
#      f.write(symptoms[i] + '\n')

symptoms_cnt = {}

for s in train_set+test_set:
    for k, v in s["goal"]["explicit_inform_slots"].items():
        if v is True:
            if k not in symptoms_cnt.keys():
                symptoms_cnt[k] = 1
            else:
                symptoms_cnt[k] += 1
    for k, v in s["goal"]["implicit_inform_slots"].items():
        if v is True:
            if k not in symptoms_cnt.keys():
                symptoms_cnt[k] = 1
            else:
                symptoms_cnt[k] += 1
print(symptoms_cnt)

sort_symptoms = sorted(symptoms_cnt.items(), key=lambda x:x[1], reverse=True)
# print(sort_symptoms)
high_freq = []
for p in sort_symptoms:
    high_freq.append(p[0])
# print(high_freq)

with open("req_dise_sym_dict.p", "wb") as f:
    pickle.dump(high_freq, f)

