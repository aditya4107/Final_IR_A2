ranking = {
    'MED-1293': 649.060601913883,
    'MED-1794': 624.2776889048488,
    'MED-957': 565.0683465235097,
    'MED-1208': 495.89532195302115,
    'MED-1159': 476.82352889334265,
    'MED-2494': 476.82352889334265,
    'MED-52': 476.82352889334265,
    'MED-1361': 426.13659577240355,
    'MED-1375': 426.13659577240355,
    'MED-1369': 406.55371151046705
}

# sorted_ranking = sorted(ranking.items(), key=lambda x: (-x[1], x[0]))

# print(sorted(key_value.items(), key=lambda kv: (-kv[1], kv[0])))

print(sorted(ranking.items(), key=lambda x: (x[0])))

# print(sorted_ranking)
