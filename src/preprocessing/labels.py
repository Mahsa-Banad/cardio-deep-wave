def map_label(symbol):
    normal = ['N']
    superventricular = ['A']
    ventricular = ['V']

    if symbol in normal:
        return 0
    elif symbol in superventricular:
        return 1
    elif symbol in ventricular:
        return 2
    else:
        return -1