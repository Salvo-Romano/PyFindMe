class CTLVocabulary:
    def __init__(self, atomic_propositions):
        # Rimossi '(' e ')' perché con max 1 operatore booleano creano solo 
        # duplicati sintattici inutili come "E F ( Alive_N )"
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
    FSM rigorosa per CTL senza tautologie né ridondanze sintattiche.
    - 'A G Alive_N' / 'E G Alive_N': Invariante di sopravvivenza (vero in s0).
    - 'E F <Goal>': Raggiungibilità esistenziale di un obiettivo tattico in depth >= 1.
    """
    def __init__(self, vocab: CTLVocabulary):
        self.vocab = vocab
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
            # Su Kripke a 1 agente, 'A F' è sempre falsificata dal ramo passivo DEFEND dell'NPC
            if self.quantifier == 'A':
                allowed = ['G']
            else:
                allowed = ['F', 'G']
            
        elif self.mode == 'FORMULA':
            if self.last_token in ['F', 'G', 'and', 'or']:
                allowed = [p for p in self.vocab.propositions if p not in self.used_props]
                
                if self.temp_op == 'G':
                    # Solo Alive_N è garantita vera in s0 anche a parità o svantaggio di HP
                    allowed = [p for p in allowed if p == 'Alive_N']
                elif self.temp_op == 'F' and self.bool_count == 0:
                    allowed = [p for p in allowed if p != 'Alive_N']
                
                if self.last_bool_op == 'or' and 'Horizon' in allowed:
                    allowed.remove('Horizon')
                if self.last_bool_op == 'or' and 'Alive_N' in allowed:
                    allowed.remove('Alive_N')

            elif self.last_token in self.vocab.propositions:
                if self.temp_op == 'G':
                    allowed = ['<EOS>']
                elif self.last_token == 'Horizon' and self.bool_count < 1:
                    allowed = ['and']
                else:
                    if self.bool_count < 1:
                        if 'Alive_N' in self.used_props:
                            allowed = ['and', '<EOS>']
                        else:
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