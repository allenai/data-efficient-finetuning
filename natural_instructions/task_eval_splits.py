'''
Splits of the evak tasks, as laid out in the NIV2 paper
'''
ENTAILMENT = [
    "task937",
    # "task202",
    # "task936",
    # "task641",
    # "task1344",
    # "task1615",
    # "task1385",
    # "task935",
    # "task199",
    # "task1388",
    # "task1554",
    # "task640",
    # "task534",
    # "task201",
    # "task1386",
    # "task463",
    # "task1387",
    # "task738",
    # "task1529",
    # "task190",
    # "task200",
    # "task1612",
    # "task970",
    # "task890",
    # "task464",
    # "task1516",
    # "task642",
]

CAUSE_EFFECT_CLASSIFICATION = [
    #"task1178",
    "task391",
    # "task939",
    # "task392",
    # "task938",
    # "task1168",
    # "task828",
    # "task1628",
    # "task943",
    # "task1182",
    # "task1171",
    # "task968",
    # "task942",
    # "task1181",
    # "task1172",
    # "task1393",
    # "task1174",
    # "task1627",
    # "task1177",
    # "task1184",
    # "task1185",
    # "task1176",
    # "task614",
    # "task1629",
    # "task1175",
    # "task827",
    # "task1173",
    # "task1180",
    # "task1170",
    # "task1183",
    # "task969",
    # "task941",
    # "task1626",
    # "task940",
    # "task393",
    # "task1169",
    # "task1179",
]

COREFERENCE = [
    "task1391",
    # "task1664",
    # "task304",
    # "task892",
    # "task891",
    # "task330",
    # "task401",
    # "task033",
    # "task133",
    # "task329",
    # "task249",
    # "task648",
    # "task1390",
    # "task893",
]

DIALOGUE_ACT_RECOGNITION = [
    "task879",
    # "task362",
    # "task1533",
    # "task1534",
    # "task880",
    # "task1531",
    # "task1394",
]

ANSWERABILITY = [
    "task020",
    # "task050",
    # "task1439",
    # "task233",
    # "task226",
    # "task396",
    # "task1640",
    # "task232",
    # "task1442",
    # "task242",
    # "task1624",
    # "task520",
    # "task290",
    # "task349",
]

WORD_ANALOGY = [
    "task1155",
    # "task1152",
    # "task1158",
    # "task1156",
    # "task1157",
    # "task1159",
    # "task1153",
    # "task1154",
]

OVERLAP = [
    "task039",
    # "task281",
]

KEYWORD_TAGGING = [
    "task613",
    # "task645",
    # "task620",
    # "task036",
    # "task623",
]

QUESTION_REWRITING = [
    "task670",
    # "task121",
    # "task1195",
    # "task442",
    # "task1345",
    # "task035",
    # "task671",
    # "task1562",
    # "task1622",
    # "task034",
    # "task402",
]

TITLE_GENERATION = [
    "task1356",
    # "task1540",
    # "task1659",
    # "task569",
    # "task1342",
    # "task220",
    # "task1561",
    # "task418",
    # "task1358",
    # "task769",
    # "task219",
    # "task602",
    # "task1586",
    # "task743",
    # "task500",
    # "task619",
    # "task510",
    # "task288",
    # "task1161",
]

DATA_TO_TEXT = [
    "task957",
    # "task1631",
    # "task1598",
    # "task1728",
    # "task102",
    # "task677",
    # "task1407",
    # "task1409",
    # "task760",
]

GRAMMAR_ERROR_CORRECTION = [
    "task1557"
]

ALL_EVAL_SPLITS = {
    "entailment": ENTAILMENT,
    "cause_effect_classification": CAUSE_EFFECT_CLASSIFICATION,
    "coreference": COREFERENCE,
    "dialogue_act_recognition": DIALOGUE_ACT_RECOGNITION,
    "answerability": ANSWERABILITY,
    "word_analogy": WORD_ANALOGY,
    "overlap": OVERLAP,
    "keyword_tagging": KEYWORD_TAGGING,
    "question_rewriting": QUESTION_REWRITING,
    "title_generation": TITLE_GENERATION,
    "data_to_text": DATA_TO_TEXT,
    "grammar_error_correction": GRAMMAR_ERROR_CORRECTION
}

for k in ALL_EVAL_SPLITS:
    print(k, len(ALL_EVAL_SPLITS[k]))