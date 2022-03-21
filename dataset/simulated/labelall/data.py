import pickle
import numpy as np
import sys
import heapq

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

goal_set = pickle.load(open("goal_dict_original.p", "rb"))
train_set = goal_set["train"]
test_set = goal_set["test"]

high_freq = pickle.load(open("req_dise_sym_dict.p", "rb"))
print(high_freq)
# new_goal_set = {"train":[], "test":[], "validate":[]}
# for goal in train_set:
#     for key, val in goal["goal"].items():
#         goal[key] = val
#     del goal["goal"]
#     new_goal_set["train"].append(goal)
# for goal in test_set:
#     for key, val in goal["goal"].items():
#         goal[key] = val
#     del goal["goal"]
#     new_goal_set["test"].append(goal)
# with open("goal_dict_original.p", "wb") as f:
#     pickle.dump(new_goal_set, f)
# sym_dict = pickle.load(open("sym_dict.p","rb"))

# dise_dict = pickle.load(open("dise_dict.p","rb"))
# sym_given_dis = np.loadtxt("sym_dise_pro.txt")
# dis_given_sym = np.loadtxt("dise_sym_pro.txt")
# slot_set = pickle.load(open("slot_set.p", "rb"))
# all_ = [None for i in range(len(slot_set.keys()))]
# for key, idx in slot_set.items():
#     if idx >= len(all_):
#         print(key, idx)
#         break
#     all_[idx] = key
# print(slot_set)
# action_mat = np.zeros([len(sym_dict)+len(dise_dict)+2, len(sym_dict)+len(dise_dict)+2])
# action_mat[0, 0] = 1
# action_mat[1, 1] = 1
# for key, idx in dise_dict.items():
#     action_mat[2+len(dise_dict):, idx+2] = dis_given_sym[idx]
#     # print(np.sum(dis_given_sym[idx]))

# for key, idx in sym_dict.items():
#     action_mat[2:2+len(dise_dict), idx+2+len(dise_dict)] = sym_given_dis[idx]
# np.savetxt("action_mat.txt", action_mat, fmt="%.8f")
    # print(np.sum(sym_given_dis[idx]))

# for i in range(action_mat.shape[1]):
#     print(np.sum(action_mat[:, i]))






# slot_set = {}
# for goal in goal_set["train"]+goal_set["test"]:
#     if goal["disease_tag"] in slot_set.keys():
#         slot_set[goal["disease_tag"]] += 1
#     else:
#         slot_set[goal["disease_tag"]] = 1
#     for symptom, val in goal["goal"]["implicit_inform_slots"].items():
#         # if symptom == "Diaper rash":
#         #     print("here")
#         #     del goal["goal"]["implicit_inform_slots"][symptom]
#         #     goal["goal"]["implicit_inform_slots"]["Diaper rash symptom"] = val
#         if val == True:
#             if symptom in slot_set.keys():
#                 slot_set[symptom] += 1
#             else:
#                 slot_set[symptom] = 1
#     for symptom, val in goal["goal"]["explicit_inform_slots"].items():
#         if val == True:
#             if symptom in slot_set.keys():
#                 slot_set[symptom] += 1
#             else:
#                 slot_set[symptom] = 1
#         # if symptom == "Diaper rash":
#         #     print("here")
#         #     del goal["goal"]["explicit_inform_slots"][symptom]
#         #     goal["goal"]["explicit_inform_slots"]["Diaper rash symptom"] = val
#     # new_goal["test"].append(goal)


# len_ = len(goal_set["train"]) + len(goal_set["test"])
# vals = np.zeros([len(sym_dict)])
# for key in symptom_prior.keys():
#     vals[sym_dict[key]] = symptom_prior[key] / len_
# print(vals)

# symptom_given_disease = {}
# req = {}

# dise_sym_num_dict =  []
# disease_cnt = {}
# vals2 = np.zeros([len(sym_dict), len(dise_dict)]) #sym_dise_pro
# vals3 = np.zeros([len(dise_dict), len(sym_dict)]) #dise_sym_pro
# for goal in goal_set["train"]+goal_set["test"]:
#     # disease = goal["disease_tag"]
#     # if disease not in symptom_given_disease.keys():
#     #     symptom_given_disease[disease] = {}
#     #     req[disease] = []
#     # else:
#     #     disease_sym_num_dict[disease] = 1

#     for symptom, val in goal["implicit_inform_slots"].items():
#         if val == True:
#             # if symptom in symptom_given_disease[disease].keys():
#             #     symptom_given_disease[disease][symptom] += 1
#             # else:
#             #     symptom_given_disease[disease][symptom] = 1

#             if symptom in symptom_cnt.keys():
#                 symptom_cnt[symptom] += 1
#             else:
#                 symptom_cnt[symptom] = 1
#     for symptom, val in goal["explicit_inform_slots"].items():
#         # if val == True:
#         #     if symptom in symptom_given_disease[disease].keys():
#         #         symptom_given_disease[disease][symptom] += 1
#         #     else:
#         #         symptom_given_disease[disease][symptom] = 1

#             if symptom in symptom_cnt.keys():
#                 symptom_cnt[symptom] += 1
#             else:
#                 symptom_cnt[symptom] = 1
# sort_symptoms = sorted(symptom_cnt.items(), key=lambda x:x[1], reverse=True)
# # print(sort_symptoms)
# high_freq = []
# for p in sort_symptoms:
#     if len(high_freq) > 90:
#         break
#     high_freq.append(p[0])
# print(high_freq)

# for disease, symps in symptom_given_disease.items():
#     sort_symptoms = sorted(symps.items(), key=lambda x: x[1], reverse=True)
#     for symptom in sort_symptoms:
#         req[disease].append(symptom[0])

# with open("req_dise_sym_dict.p", "wb") as f:
#     pickle.dump(req, f)
# # print(symptom_cnt)
# for key, val in symptom_given_disease.items():
#     for symp, cnt in val.items():
#         vals2[sym_dict[symp]][dise_dict[key]] = cnt / disease_cnt[key]
#         vals3[dise_dict[key]][sym_dict[symp]] = cnt / symptom_cnt[symp]

# # print(vals3)
# np.savetxt("sym_prio.txt", vals, fmt="%.8f")
# np.savetxt("sym_dise_pro.txt", vals2, fmt="%.8f")
# np.savetxt("dise_sym_pro.txt", vals3, fmt="%.8f")

# sym_dict = {}
# dise_dict = {}
# cnt = 0

# cnt = 0
# with open("diseases.txt", "r") as f:
#     for line in f.readlines():
#         dise_dict[line.rstrip('\n')] = cnt
#         cnt += 1

# with open("sym_dict.p", "wb") as f:
#     pickle.dump(sym_dict, f)

# with open("dise_dict.p", "wb") as f:
#     pickle.dump(dise_dict, f)

"""
disease_set = {}
cnt = 0
for k in disease_symp.keys():
    disease_set[k] = cnt
    cnt += 1

with open("disease_set.p", "wb") as f:
    pickle.dump(disease_set, f)


group_idx = {'1':0, '4':1, '5':2, '6':3, '7':4, '12': 5, '13': 6, '14':7, '19':8}

"""
