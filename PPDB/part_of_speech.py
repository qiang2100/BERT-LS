class PartOfSpeech:
    NOUN = 'noun'
    VERB = 'verb'
    ADJECTIVE = 'adjective'
    ADVERB = 'adverb'

    pos2con = {
        'NN': ['NN'],
        'NNS': ['NNS'],
        'NNP': ['NNP'],
        'NNPS': ['NNPS'],
        'NP': ['NP'],
        'VB': ['VB'],
        'VBD': ['VBD'],
        'VBG': ['VBG'],
        'VBN': ['VBN'],
        'VBZ': ['VBZ'],
        'VBP': ['VBP'],
        'a': ['JJ', 'JJR', 'JJS', 'IN'],
        's': ['JJ', 'JJR', 'JJS', 'IN'],  # Adjective Satellite
        'r': ['RB', 'RBR', 'RBS'],  # Adverb
    }

    con2pos = {}
    poses = []
    for key, values in pos2con.items():
        poses.extend(values)
        for value in values:
            if value not in con2pos:
                con2pos[value] = []
            con2pos[value].append(key)

    @staticmethod
    def pos2constituent(pos):
        if pos in PartOfSpeech.pos2con:
            return PartOfSpeech.pos2con[pos]
        return []

    @staticmethod
    def constituent2pos(con):
        if con in PartOfSpeech.con2pos:
            return PartOfSpeech.con2pos[con]
        return []

    @staticmethod
    def get_pos():
        return PartOfSpeech.poses
