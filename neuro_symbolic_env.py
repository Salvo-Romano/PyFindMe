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
    Genera solo formule canoniche: [E/A] [F/G] Prop1 ([and/or] Prop2) <EOS>
    """
    def __init__(self, vocab: CTLVocabulary):
        self.vocab = vocab
        self.reset()
        
    def reset(self):
        self.mode = 'START'
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
            allowed = ['F', 'G']
            
        elif self.mode == 'FORMULA':
            if self.last_token in ['F', 'G', 'and', 'or']:
                # Niente ripetizioni della stessa proposizione
                allowed = [p for p in self.vocab.propositions if p not in self.used_props]
                
                # REGOLA SEMANTICA DI 'G' (Invariante):
                # 'G phi' richiede che phi sia GIA' vera nello stato corrente.
                # Dead_P, LowHP_P, SpecialReady_N e Horizon sono quasi sempre falsi a inizio match:
                # usarli come radice di 'G' causa il 90% delle Sconfitte Tecniche a Turno 1.
                if self.temp_op == 'G' and self.bool_count == 0:
                    allowed = [p for p in allowed if p in ['Alive_N', 'HP_Advantage_N']]
                elif self.temp_op == 'G' and 'Horizon' in allowed:
                    allowed.remove('Horizon')
                
                if self.last_bool_op == 'or' and 'Horizon' in allowed:
                    allowed.remove('Horizon')
                
                if self.last_bool_op == 'or' and 'Alive_N' in allowed:
                    allowed.remove('Alive_N')

            elif self.last_token in self.vocab.propositions:
                if self.last_token == 'Horizon' and self.bool_count < 1:
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