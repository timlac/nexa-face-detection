

def save_data(data, path):
    with open(path + '.pckl', 'wb') as fil:
        import pickle
        pickle.dump(data, fil)
