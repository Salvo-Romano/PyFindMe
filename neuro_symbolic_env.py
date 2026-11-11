class CTLVocabulary:
    def __init__(self, atomic_propositions):
        self.operators = ['E', 'A', 'F', 'G', 'and', 'or']
        self.propositions = atomic_propositions
        self.eos_token = '<EOS>'
        
        self.vocab = self.operators + self.propositions + [self.eos_token]
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}
        
    def get_size(self):
        return len(self.vocab)


class GrammarMasker:
    """
    FSM rigorosa per CTL senza Alive_N.
    - A G / E G: mantenimento di invarianti tattici (HP_Advantage_N, Fortified_N, CounterReady_N)
    - E F: sintesi verso obiettivi primari (Dead_P, CanKill_N, LowHP_P, HP_Advantage_N, SpecialReady_N, Fortified_N)
    """
    def __init__(self, vocab: CTLVocabulary):
        self.vocab = vocab
        
        self.primary_goals = [
            'Dead_P', 'CanKill_N', 'LowHP_P', 
            'HP_Advantage_N', 'SpecialReady_N', 'Fortified_N'
        ]
        self.invariants = ['HP_Advantage_N', 'Fortified_N', 'CounterReady_N']
        self.qualifiers = ['Threat_P', 'CounterReady_N', 'Fortified_N']
        self.force_and = ['Horizon']
        
        self.reset()
        
    def reset(self):
        self.mode = 'START'
        self.quantifier = None
        self.last_token = None
        self.temp_op = None
        self.bool_count = 0
        self.last_bool_op = None
        self.used_props = set()
        
    def _create_mask(self, allowed_tokens):
        mask = [0] * self.vocab.get_size()
        for t in allowed_tokens:
            if t in self.vocab.token2id:
                mask[self.vocab.token2id[t]] = 1
        return mask

    def get_valid_mask(self):
        allowed = []
        
        if self.mode == 'START':
            allowed = ['E', 'A']
            
        elif self.mode == 'TEMP_OP':
            if self.quantifier == 'A':
                allowed = ['G']
            else:
                allowed = ['F', 'G']
            
        elif self.mode == 'FORMULA':
            if self.last_token in ['F', 'G', 'and', 'or']:
                available = [p for p in self.vocab.propositions if p not in self.used_props]
                
                if self.temp_op == 'G':
                    allowed = [p for p in available if p in self.invariants]
                elif self.temp_op == 'F' and self.bool_count == 0:
                    allowed = [p for p in available if p in self.primary_goals]
                elif self.last_token == 'and':
                    allowed = [p for p in available]
                elif self.last_token == 'or':
                    allowed = [p for p in available if p in self.primary_goals]

            elif self.last_token in self.vocab.propositions:
                if self.temp_op == 'G':
                    allowed = ['<EOS>']
                elif self.last_token in self.force_and:
                    allowed = ['and'] if self.bool_count < 1 else ['<EOS>']
                else:
                    if self.bool_count < 1:
                        allowed = ['and', 'or', '<EOS>']
                    else:
                        allowed = ['<EOS>']
                        
        mask = self._create_mask(allowed)
        if sum(mask) == 0:
            return self._create_mask(['<EOS>'])
            
        return mask

    def update_state(self, token_id):
        token = self.vocab.id2token[token_id]
        
        if self.mode == 'START':
            self.quantifier = token
            self.mode = 'TEMP_OP'
        elif self.mode == 'TEMP_OP':
            self.temp_op = token
            self.mode = 'FORMULA'
        elif self.mode == 'FORMULA':
            if token in ['and', 'or']:
                self.bool_count += 1 
                self.last_bool_op = token
            elif token in self.vocab.propositions:
                self.used_props.add(token)
                
        self.last_token = token
        return token