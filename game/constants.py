import numpy as np

EDGE_SOURCES = np.array([
# 0: Bombay       | 1: Hyderabad      | 2: Madras         | 3: Seringapatam   | 4: Coimbatore
0, 0, 0,            1, 1, 1, 1,         2, 2, 2, 2,         3, 3, 3, 3,         4, 4, 4, 4,
# 5: Satara       | 6: Raichur        | 7: Masulipatam    | 8: Goa            | 9: Darwar
5, 5, 5, 5,         6, 6, 6, 6,         7, 7,               8, 8, 8,            9, 9, 9,
# 10: Anantapur   | 11: Chitaldoorg   | 12: Mangalore     | 13: Bangalore     | 14: Vellore
10, 10, 10, 10,     11, 11, 11, 11,     12, 12, 12,         13, 13, 13,         14, 14, 14, 14,
# 15: Mahé        | 16: Pondicherry   | 17: Erode         | 18: Trichy        | 19: Alwaye
15, 15,             16, 16, 16,         17, 17, 17, 17, 17, 18, 18, 18, 18,     19, 19, 19,
# 20: Dindigul    | 21: Ramnad        | 22: Travancore    | 23: Ceylon        | 24: Poona
20, 20, 20,         21, 21, 21, 21,     22, 22, 22,         23, 23, 23,         24, 24, 24
])

EDGE_DESTS = np.array([
# 0: Bombay       | 1: Hyderabad      | 2: Madras         | 3: Seringapatam   | 4: Coimbatore
5, 8, 24,           6, 7, 10, 24,       7, 10, 14, 16,      12, 13, 15, 17,     15, 17, 19, 20,
# 5: Satara       | 6: Raichur        | 7: Masulipatam    | 8: Goa            | 9: Darwar
0, 6, 9, 24,        1, 5, 10, 11,       1, 2,               0, 9, 12,           5, 8, 11,
# 10: Anantapur   | 11: Chitaldoorg   | 12: Mangalore     | 13: Bangalore     | 14: Vellore
1, 2, 6, 14,        6, 9, 12, 13,       3, 8, 11,           3, 11, 14,          2, 10, 13, 17,
# 15: Mahé        | 16: Pondicherry   | 17: Erode         | 18: Trichy        | 19: Alwaye
3, 4,               2, 17, 18,          3, 4, 14, 16, 18,   16, 17, 20, 23,     4, 21, 22,
# 20: Dindigul    | 21: Ramnad        | 22: Travancore    | 23: Ceylon        | 24: Poona
4, 18, 21,          19, 20, 22, 23,     19, 21, 23,         18, 21, 22,         0, 1, 5
])

INDEX_MAP = {
    0: "Bombay",
    1: "Hyderabad",
    2: "Madras",
    3: "Seringapatam",
    4: "Coimbatore",
    5: "Satara",
    6: "Raichur",
    7: "Masulipatam",
    8: "Goa",
    9: "Darwar",
    10: "Anantapur",
    11: "Chitaldoorg",
    12: "Mangalore",
    13: "Bangalore",
    14: "Vellore",
    15: "Mahé",
    16: "Pondicherry",
    17: "Erode",
    18: "Trichy",
    19: "Alwaye",
    20: "Dindigul",
    21: "Ramnad",
    22: "Travancore",
    23: "Ceylon",
    24: "Poona"
}

CARD_VALUE = np.array([3, 2, 2, 1, 1, 1])

EDGES = len(EDGE_SOURCES)

NODES = len(INDEX_MAP)

CARDS = len(CARD_VALUE)

TURNS = 4

NO_UNIT = -1

ADJACENCY_MATRIX = np.zeros((NODES, NODES), dtype = bool)
ADJACENCY_MATRIX[EDGE_SOURCES, EDGE_DESTS] = True

# (Bombay, Hyderabad, Madras, Seringapatam, Coimbatore)
KEYS = np.array([
    True,  True,  True,  True,  True,  False, False, False, 
    False, False, False, False, False, False, False, False, 
    False, False, False, False, False, False, False, False, False
], dtype=bool)
assert len(KEYS) == NODES

# (Bombay, Madras, Masulipatam, Goa, Mangalore, Mahé, Pondicherry, Ramnad, Travancore, Ceylon)
COASTAL = np.array([
    True,  False, True,  False, False, False, False, True,  
    True,  False, False, False, True,  False, False, True,  
    True,  False, False, False, False, True,  True,  True, False
], dtype=bool)
assert len(COASTAL) == NODES

KEY_INDICES = np.where(KEYS)[0]

COASTAL_INDICES = np.where(COASTAL)[0]

MOVE_SPACE = [
    ("Move", EDGES, "edge"),
    ("Tire", NODES, "node"),
    ("Sepoy Mutiny", NODES, "node"),
    ("French Alliance", NODES, "node"),
    ("Monsoon", NODES, "node"),
    ("Cavalry Raid", 1, "blank"),
    ("Sea Trade", NODES*len(COASTAL_INDICES), "coastal"),
    ("Mysore Power", CARDS, "mcard"),
    ("Draw Iron Rockets", CARDS, "mcard"),
    ("Draw Sepoy Mutiny", CARDS, "mcard"),
    ("Draw French Alliance", CARDS, "mcard"),
    ("Pass Mysore", 1, "blank"),
    ("Highlanders", NODES, "node"),
    ("Royal Navy", NODES*len(COASTAL_INDICES), "coastal"),
    ("Divide and Rule", EDGES, "edge"),
    ("Force March", EDGES, "edge"),
    ("Princely States", NODES, "node"),
    ("British Power", CARDS, "bcard"),
    ("Draw Wall Breach", CARDS, "bcard"),
    ("Draw Highlanders", CARDS, "bcard"),
    ("Draw Royal Navy", CARDS, "bcard"),
    ("Pass British", 1, "blank")
]

WHO_TO_MOVE = ["British Move", "Mysore Card", "British Card"]

MYSORE_CARDS = ["Iron Rockets", "Sepoy Mutiny", "French Alliance", "Monsoon", "Cavalry Raid", "Sea Trade"]
BRITISH_CARDS = ["Wall Breach", "Highlanders", "Royal Navy", "Divide and Rule", "Force March", "Princely States"]

NODE_TO_IDX = {value: i for i, (_, value) in enumerate(INDEX_MAP.items())}
WHO_TO_MOVE_TO_IDX = {name: i for i, name in enumerate(WHO_TO_MOVE)}
MYSORE_CARDS_TO_IDX = {name: i for i, name in enumerate(MYSORE_CARDS)}
BRITISH_CARDS_TO_IDX = {name: i for i, name in enumerate(BRITISH_CARDS)}

GAME_VECTOR_LENGTH = NODES * 5 + CARDS * 2 + TURNS + 3 + 4

BRITISH_MOVES_SPACE = sum(val for _, val, _ in MOVE_SPACE[:2])
MYSORE_CARDS_SPACE = sum(val for _, val, _ in MOVE_SPACE[2:12])
BRITISH_CARDS_SPACE = sum(val for _, val, _ in MOVE_SPACE[12:])

MOVE_VECTOR_LENGTH = BRITISH_MOVES_SPACE + MYSORE_CARDS_SPACE + BRITISH_CARDS_SPACE

DEFAULT_MODEL = "ai/models/alphatigerv13.pt"
DEFAULT_SIMS = 400

CARDS_ABBREV = {
    "Iron Rockets": "IR",
    "Sepoy Mutiny": "SM",
    "French Alliance": "FA",
    "Monsoon": "MS",
    "Cavalry Raid": "CR",
    "Sea Trade": "ST",
    "Wall Breach": "WB",
    "Highlanders": "HL",
    "Royal Navy": "RN",
    "Divide and Rule": "DR",
    "Force March": "FM",
    "Princely States": "PS",
    "Draw Iron Rockets": "IR",
    "Draw Sepoy Mutiny": "SM",
    "Draw French Alliance": "FA",
    "Draw Wall Breach": "WB",
    "Draw Highlanders": "HL",
    "Draw Royal Navy": "RN"
}

NODES_ABBREV = {
    0: "bom", 1: "hyd", 2: "mad", 3: "srp", 4: "cbt",
    5: "sat", 6: "rch", 7: "msp", 8: "goa", 9: "dwr",
    10: "ant", 11: "ctd", 12: "mlr", 13: "blr", 14: "vlr",
    15: "mhe", 16: "pdc", 17: "erd", 18: "tri", 19: "alw",
    20: "dng", 21: "rmd", 22: "trv", 23: "cyl", 24: "pna"
}