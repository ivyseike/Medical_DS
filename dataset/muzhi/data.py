import pickle
import numpy as np
import sys

def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """

    slot_set = {}
    with open(path, 'r') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set

goal_set = pickle.load(open("goal_dict_original.p", "rb"))
for goal in goal_set["train"]:
    print(goal)
# train_set = goal_set["train"]
# test_set = goal_set["test"]

# dise_given_sym = np.loadtxt("dise_sym_pro.txt")
# print(dise_given_sym.shape)
# for i in range(dise_given_sym.shape[1]):
#     print(np.sum(dise_given_sym[:, i]))
"""
all_val = 0
for g in train_set:
    for key, val in g["implicit_inform_slots"].items():
        if val == True:
            all_val += 1
    for key, val in g["explicit_inform_slots"].items():
        if val == True:
            all_val += 1
    #all_val += len(g["implicit_inform_slots"])
    #all_val += len(g["explicit_inform_slots"])

#print(all_val)
"""


#sym_dict = text_to_dict("symptoms.txt")
#print(sym_dict)

# diaacts = text_to_dict("dia_acts.txt")
# print(diaacts)
# req_dise_sym_dict = pickle.load(open("req_dise_sym_dict.p", "rb"))
# dise_sym_num_dict = pickle.load(open("dise_sym_num_dict.p","rb"))
# action_mat = np.loadtxt("action_mat.txt")
# np.set_printoptions(threshold=sys.maxsize)
# for i in range(72):
#     print(np.sum(action_mat[i, :]))


# print(req_dise_sym_dict)
# print(dise_sym_num_dict)
#sp = np.loadtxt("sym_prio.txt")
#print(sp)

#disease_symp = pickle.load(open("disease_symptom.p", "rb"))
#slot_set = text_to_dict("slot_set.txt")
#print(slot_set)
# sys_request_slots = ['普通感冒', '干咳', '咳嗽', '厌食', '发热', '上呼吸道感染', '中等度热', '出汗', '高热', '头痛', '咽喉不适', '低热', '呕吐', '精神软', '鼻流涕', '喷嚏', '鼻塞', '四肢厥冷', '急性气管支气管炎', '咳痰', '稀便', '食欲不佳', '腹痛', '恶心', '干呕', '肠炎', '过敏', '有痰', '痰鸣音', '扁桃体炎', '退热', '支气管炎', '大便酸臭', '消化不良', '腹泻', '气管炎', '肺炎', '血便', '皮疹', '咽喉炎', '喘息', '水样便', '食欲不振', '绿便', '肛门红肿', '支气管肺炎', '口臭', '哭闹', '湿疹', '鼻炎', '病毒感染', '睡眠障碍', '反复发热', '嗜睡', '便秘', '贫血', '大便粘液', '粗糙呼吸音', '腹胀', '屁', '沙哑', '细菌感染', '尿量减少', '腹部不适', '肠鸣音', '支原体感染']
#print(sys_request_slots)
# disease_set = pickle.load(open("disease_set.p", "rb"))
#disease_set = {}
#cnt = 0

# symptom_prior = {}
# all_cnt = 0
# for key, val in dise_sym_num_dict.items():
#     for d, cnt in val.items():
#         if d in symptom_prior.keys():
#             symptom_prior[d] += cnt
#         else:
#             symptom_prior[d] = cnt
#         all_cnt += cnt
#print(len(train_set))

#print(symptom_prior)
# res = []
# for dis in sys_request_slots:
#     res += [symptom_prior[dis] / (len(train_set)+len(test_set))]
#print(res)
#print(len(symptom_prior))
"""

for k in disease_symp.keys():
    disease_set[k] = cnt
    cnt += 1
print(disease_set)


#with open("disease_set.p", "wb") as f:
#    pickle.dump(disease_set, f)

#group_idx = {'1':0, '4':1, '5':2, '6':3, '7':4, '12': 5, '13': 6, '14':7, '19':8}
#arr = np.zeros([9, len(slot_set)-1])
#sym_disease = np.zeros([len(slot_set)-1, 9])
#prior_sym = np.zeros([len(slot_set)-1])

for s in train_set:
    for k, v in s["goal"]["explicit_inform_slots"].items():
        if v is True:
            arr[group_idx[s["group_id"]]][slot_set[k]] += 1
            sym_disease[slot_set[k]][group_idx[s["group_id"]]] += 1
            prior_sym[slot_set[k]] += 1
    for k, v in s["goal"]["implicit_inform_slots"].items():
        if v is True:
            arr[group_idx[s["group_id"]]][slot_set[k]] += 1
            sym_disease[slot_set[k]][group_idx[s["group_id"]]] += 1
            prior_sym[slot_set[k]] += 1

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)
sum_ = np.sum(arr, axis=1)
sum_2 = np.sum(sym_disease, axis=1)
sum_3 = np.sum(prior_sym, axis=0)
# print(sum_3)
# print(sum_)
# print(sum_2)
# print(arr)
# print(sym_disease)
for i in range(9):
    arr[i] /= sum_[i]
for i in range(len(slot_set)-1):
    sym_disease[i] /= sum_2[i]
    prior_sym[i] /= sum_3
# print(arr)
# print(sym_disease)
# print(prior_sym)

np.savetxt("prior_symptom.txt", prior_sym, fmt="%.8f")
np.savetxt("symptom_given_disease.txt", arr, fmt="%.8f")
np.savetxt("disease_given_symptom.txt", sym_disease, fmt="%.8f")
"""

