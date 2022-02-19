import pickle
import copy
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys_inform_slots = ['disease']

start_dia_acts = {
    'request': ['disease']
}


# dxy
'''
sys_inform_slots_values = ['小儿腹泻', '小儿手足口病', '过敏性鼻炎', '上呼吸道感染']
sys_request_slots = ['稀便', '厌食', '精神萎靡', '尿少', '发热', '烦躁不安', '疱疹', '咽部不适', '淋巴结肿大', '鼻塞', '咳嗽', '抽动', '皮疹', '流涎', '咳痰', '喷嚏', '流涕', '绿便', '腹痛', '肠鸣音亢进', '呕吐', '盗汗', '呼吸困难', '肛门排气增加', '反胃', '蛋花样便', '腹胀', '过敏', '鼻痒', '呼吸音粗', '头痛', '鼻衄', '眼部发痒', '臭味', '舌苔发白', '口渴', '畏寒', '嗳气', '体重减轻']
sys_request_slots_highfreq = ['稀便', '呕吐', '发热', '烦躁不安', '厌食', '精神萎靡', '绿便', '流涕', '蛋花样便', '腹痛', '皮疹', '疱疹', '咽部不适', '流涎', '咳嗽', '鼻塞', '过敏', '呼吸困难', '喷嚏', '抽动', '咳痰', '鼻痒']
'''


# muzhi

# sys_inform_slots_values = ['上呼吸道感染', '小儿支气管炎', '小儿腹泻', '小儿消化不良']
# sys_request_slots_highfreq = ['发热', '咳嗽', '鼻流涕', '普通感冒', '中等度热', '有痰', '鼻塞', '低热', '喷嚏', '呕吐', '支气管炎', '痰鸣音', '咳痰', '急性气管支气管炎', '腹泻', '稀便', '水样便', '消化不良', '绿便', '血便', '大便粘液', '屁', '哭闹', '厌食']
# sys_request_slots = ['普通感冒', '干咳', '咳嗽', '厌食', '发热', '上呼吸道感染', '中等度热', '出汗', '高热', '头痛', '咽喉不适', '低热', '呕吐', '精神软', '鼻流涕', '喷嚏', '鼻塞', '四肢厥冷', '急性气管支气管炎', '咳痰', '稀便', '食欲不佳', '腹痛', '恶心', '干呕', '肠炎', '过敏', '有痰', '痰鸣音', '扁桃体炎', '退热', '支气管炎', '大便酸臭', '消化不良', '腹泻', '气管炎', '肺炎', '血便', '皮疹', '咽喉炎', '喘息', '水样便', '食欲不振', '绿便', '肛门红肿', '支气管肺炎', '口臭', '哭闹', '湿疹', '鼻炎', '病毒感染', '睡眠障碍', '反复发热', '嗜睡', '便秘', '贫血', '大便粘液', '粗糙呼吸音', '腹胀', '屁', '沙哑', '细菌感染', '尿量减少', '腹部不适', '肠鸣音', '支原体感染']

# #simulated
#疾病
sys_inform_slots_values = ['Cat scratch disease', 'Dengue fever', 'Gas gangrene', 'Chickenpox', 'Granuloma inguinale', 'Chagas disease', 'Chancroid', 'Chlamydia', 'Acariasis', 'Gonorrhea', 'Fluid overload', 'Diabetic ketoacidosis', 'Amyloidosis', 'Diabetes insipidus', 'Diabetic retinopathy', 'Diabetic peripheral neuropathy', 'Carcinoid syndrome', 'Graves disease', 'Cushing syndrome', 'Cystic Fibrosis', 'Conversion disorder', 'Chronic pain disorder', 'Acute stress reaction', 'Factitious disorder', 'Alcohol intoxication', 'Eating disorder', 'Anxiety', 'Dissociative disorder', 'Drug abuse cocaine', 'Adjustment reaction', 'Cerebral edema', 'Degenerative disc disease', 'Guillain Barre syndrome', 'Complex regional pain syndrome', 'Amyotrophic lateral sclerosis ALS', 'Encephalitis', 'Carpal tunnel syndrome', 'Extrapyramidal effect of drugs', 'Essential tremor', 'Alzheimer disease', 'Conjunctivitis due to allergy', 'Ectropion', 'Endophthalmitis', 'Cyst of the eyelid', 'Chalazion', 'Corneal disorder', 'Conductive hearing loss', 'Central retinal artery or vein occlusion', 'Acute glaucoma', 'Aphakia', 'Erythema multiforme', 'Dyshidrosis', 'Actinic keratosis', 'Dermatitis due to sun exposure', 'Eczema', 'Contact dermatitis', 'Diaper rash', 'Acne', 'Acanthosis nigricans', 'Decubitus ulcer', 'Fibromyalgia', 'Connective tissue disorder', 'Ganglion cyst', 'Ankylosing spondylitis', 'De Quervain disease', 'Chronic back pain', 'Gout', 'Flat feet', 'Adhesive capsulitis of the shoulder', 'Chondromalacia of the patella', 'Cystitis', 'Epididymitis', 'Acute kidney injury', 'Endometriosis', 'Erectile dysfunction', 'Endometrial cancer', 'Endometrial hyperplasia', 'Fibrocystic breast disease', 'Female infertility of unknown cause', 'Chronic kidney disease', 'Air embolism', 'Fat embolism', 'Drug reaction', 'Carbon monoxide poisoning', 'Fracture of the pelvis', 'Fracture of the rib', 'Allergy', 'Concussion', 'Epidural hemorrhage', 'Corneal abrasion']
#症状
sys_request_slots = ['Foot or toe stiffness or tightness', 'Disturbance of smell or taste', 'Hoarse voice', 'Congestion in chest', 'Vaginal pain', 'Leg cramps or spasms', 'Frequent urination', 'Hand or finger stiffness or tightness', 'Neck cramps or spasms', 'Nightmares', 'Symptoms of eye', 'Nose deformity', 'Difficulty in swallowing', 'Spots or clouds in vision', 'Vomiting blood', 'Skin pain', 'Lack of growth', 'Pain during pregnancy', 'Diminished vision', 'Sleepiness', 'Paresthesia', 'Hot flashes', 'Rib pain', 'Ache all over', 'Swollen or red tonsils', 'Fluid in ear', 'Pus in urine', 'Scanty menstrual flow', 'Skin irritation', 'Fatigue', 'Abnormal movement of eyelid', 'Abnormal size or shape of ear', 'Knee stiffness or tightness', 'Elbow weakness', 'Retention of urine', 'Sore throat', 'Leg pain', 'Sweating', 'Emotional symptoms', 'Leg weakness', 'Eye deviation', 'Hip pain', 'Painful sinuses', 'White discharge from eye', 'Irregular heartbeat', 'Symptoms of prostate', 'Back cramps or spasms', 'Irregular appearing scalp', 'Redness in ear', 'Acne or pimples', 'Loss of sex drive', 'Too little hair', 'Muscle swelling', 'Abnormal involuntary movements', 'Itchy eyelid', 'Knee weakness', 'Groin mass', 'Lacrimation', 'Penis redness', 'Pain during intercourse', 'Muscle weakness', 'Knee pain', 'Shoulder pain', 'Spotting or bleeding during pregnancy', 'Swollen eye', 'Allergic reaction', 'Foot or toe swelling', 'Skin growth', 'Low self-esteem', 'Diarrhea', 'Mass in scrotum', 'Swelling of scrotum', 'Suprapubic pain', 'Eye burns or stings', 'Vulvar sore', 'Ankle swelling', 'Wrist lump or mass', 'Penis pain', 'Diminished hearing', 'Difficulty eating', 'Irritable infant', 'Ear pain', 'Wrinkles on skin', 'Bones are painful', 'Muscle pain', 'Abusing alcohol', 'Temper problems', 'Back weakness', 'Focal weakness', 'Arm swelling', 'Cough', 'Blindness', 'Muscle stiffness or tightness', 'Arm weakness', 'Feeling hot and cold', 'Hysterical behavior', 'Vaginal bleeding after menopause', 'Problems with movement', 'Knee swelling', 'Increased heart rate', 'Lip swelling', 'Hip stiffness or tightness', 'Warts', 'Hemoptysis', 'Joint pain', 'Incontinence of stool', 'Lump or mass of breast', 'Double vision', 'Bleeding or discharge from nipple', 'Weight gain', 'Heavy menstrual flow', 'Irregular belly button', 'Neck stiffness or tightness', 'Symptoms of the face', 'Low back pain', 'Joint stiffness or tightness', 'Sharp chest pain', 'Impotence', 'Elbow pain', 'Lower abdominal pain', 'Ankle pain', 'Shoulder swelling', 'Difficulty speaking', 'Excessive urination at night', 'Throat swelling', 'Premature ejaculation', 'Skin lesion', 'Swollen abdomen', 'Hand or finger pain', 'Headache', 'Facial pain', 'Drug abuse', 'Intermenstrual bleeding', 'Eye redness', 'Bleeding from eye', 'Hurts to breath', 'Ringing in ear', 'Long menstrual periods', 'Shoulder stiffness or tightness', 'Infant spitting up', 'Skin rash', 'Cross-eyed', 'Bladder mass', 'Thirst', 'Foreign body sensation in eye', 'Back pain', 'Chest tightness', 'Fears and phobias', 'Feet turned in', 'Unusual color or odor to urine', 'Dizziness', 'Painful urination', 'Back stiffness or tightness', 'Diaper rash symptom', 'Arm lump or mass', 'Difficulty breathing', 'Hand or finger weakness', 'Nasal congestion', 'Sharp abdominal pain', 'Vomiting', 'Excessive appetite', 'Sneezing', 'Stiffness all over', 'Kidney mass', 'Painful menstruation', 'Skin moles', 'Eyelid swelling', 'Depressive or psychotic symptoms', 'Low back weakness', 'Skin on leg or foot looks infected', 'Foot or toe pain', 'Lower body pain', 'Restlessness', 'Ankle stiffness or tightness', 'Muscle cramps, contractures, or spasms', 'Elbow cramps or spasms', 'Knee lump or mass', 'Dry or flaky scalp', 'Eye strain', 'Hand or finger swelling', 'Excessive growth', 'Wrist swelling', 'Plugged feeling in ear', 'Unpredictable menstruation', 'Swollen tongue', 'Hand or finger cramps or spasms', 'Itching of scrotum', 'Decreased appetite', 'Polyuria', 'Skin dryness, peeling, scaliness, or roughness', 'Fever', 'Fluid retention', 'Lymphedema', 'Arm pain', 'Mass on eyelid', 'Wrist weakness', 'Peripheral edema', 'Vaginal itching', 'Blood in urine', 'Back mass or lump', 'Loss of sensation', 'Involuntary urination', 'Disturbance of memory', 'Infrequent menstruation', 'Chills', 'Itchy scalp', 'Feeling cold', 'Nailbiting', 'Burning abdominal pain', 'Absence of menstruation', 'Pulling at ears', 'Arm stiffness or tightness', 'Pain or soreness of breast', 'Pelvic pain', 'Delusions or hallucinations', 'Hand or finger lump or mass', 'Pus in sputum', 'Penile discharge', 'Leg swelling', 'Frequent menstruation', 'Decreased heart rate', 'Seizures', 'Abnormal appearing skin', 'Pain in eye', 'Vaginal discharge', 'Hostile behavior', 'Drainage in throat', 'Mass on vulva', 'Apnea', 'Bowlegged or knock-kneed', 'Neck pain', 'Skin swelling', 'Weakness', 'Shoulder cramps or spasms', 'Symptoms of bladder', 'Pain in testicles', 'Anxiety and nervousness', 'Excessive anger', 'Problems during pregnancy', 'Fainting', 'Insomnia', 'Poor circulation', 'Cramps and spasms', 'Shortness of breath', 'Side pain', 'Depression', 'Groin pain', 'Wrist stiffness or tightness', 'Symptoms of the kidneys', 'Sinus congestion', 'Infertility', 'Itching of skin', 'Palpitations', 'Frontal headache', 'Nausea', 'Foot or toe lump or mass', 'Mouth ulcer', 'Neck swelling', 'Skin on arm or hand looks infected', 'Unwanted hair', 'Wrist pain', 'Feeling ill', 'Eyelid lesion or rash', 'Coryza', 'Itchiness of eye', 'Bleeding gums']
sys_request_slots_highfreq = ['Skin rash', 'Pain in eye', 'Back pain', 'Depressive or psychotic symptoms', 'Diminished vision', 'Headache', 'Leg pain', 'Skin lesion', 'Fatigue', 'Wrist pain', 'Dizziness', 'Sharp abdominal pain', 'Loss of sensation', 'Depression', 'Anxiety and nervousness', 'Itching of skin', 'Fever', 'Vomiting', 'Nausea', 'Eye redness', 'Abnormal appearing skin', 'Sharp chest pain', 'Ankle pain', 'Neck pain', 'Excessive anger', 'Ache all over', 'Symptoms of eye', 'Pain or soreness of breast', 'Abnormal involuntary movements', 'Paresthesia', 'Shortness of breath', 'Facial pain', 'Problems with movement', 'Foot or toe pain', 'Knee lump or mass', 'Knee pain', 'Shoulder cramps or spasms', 'Low back pain', 'Joint stiffness or tightness', 'Pain during pregnancy', 'Cough', 'Shoulder pain', 'Weakness', 'Itchiness of eye', 'Disturbance of memory', 'Swollen eye', 'Arm pain', 'Hand or finger pain', 'Excessive urination at night', 'Allergic reaction', 'Hip pain', 'Lacrimation', 'Skin swelling', 'Skin growth', 'Painful urination', 'Skin dryness, peeling, scaliness, or roughness', 'Acne or pimples', 'Groin pain', 'Mass on eyelid', 'Frequent urination', 'Seizures', 'Side pain', 'Spots or clouds in vision', 'Weight gain', 'Peripheral edema', 'Abusing alcohol', 'Difficulty speaking', 'Sore throat', 'Pelvic pain', 'Diminished hearing', 'Foreign body sensation in eye', 'Difficulty in swallowing', 'Vaginal discharge', 'Penile discharge', 'Vaginal pain', 'Infertility', 'Involuntary urination', 'Vaginal bleeding after menopause', 'Diarrhea', 'Swelling of scrotum', 'Pain in testicles', 'Eyelid lesion or rash', 'Leg weakness', 'Delusions or hallucinations', 'Lip swelling', 'Insomnia', 'Joint pain', 'Retention of urine', 'Symptoms of the kidneys', 'Abnormal movement of eyelid', 'Blood in urine']

################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = 0


################################################################################
#  Diagnosis
################################################################################
NO_DECIDE = 0
NO_MATCH = "no match"
NO_MATCH_BY_RATE = "no match by rate"

################################################################################
#  Special Slot Values
################################################################################
I_AM_NOT_SURE = -1
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"

################################################################################
#  Slot Values
################################################################################
TRUE = 1
FALSE = -1
NOT_SURE = -2
NOT_MENTION = 0

################################################################################
#  Constraint Check
################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

################################################################################
#  NLG Beam Search
################################################################################
nlg_beam_size = 10

################################################################################
#  run_mode: 0 for dia-act; 1 for NL; 2 for no output
################################################################################
run_mode = 0
auto_suggest = 0

################################################################################
#   A Basic Set of Feasible actions to be Consdered By an RL agent
################################################################################
feasible_actions = [

    ############################################################################
    #   thanks actions
    ############################################################################
    {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
    {'diaact': "inform", 'inform_slots': { 'disease': 'UNK', 'taskcomplete': "PLACEHOLDER"}, 'request_slots': {} }

]

disease_actions = []
############################################################################
#   Adding the inform actions
############################################################################
for slot_val in sys_inform_slots_values:
    slot = 'disease'
    feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:slot_val, 'taskcomplete': "PLACEHOLDER"}, 'request_slots':{}})
    disease_actions.append(feasible_actions[-1])
############################################################################
#   Adding the request actions
############################################################################
symptom_actions = []
for slot in sys_request_slots:
    feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: 'UNK'}})
    symptom_actions.append(feasible_actions[-1])

