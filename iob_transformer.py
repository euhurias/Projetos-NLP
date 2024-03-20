import pandas as pd
import re
from nltk import word_tokenize, RegexpTokenizer

class iob_transformer():
    def __init__(self, coluna_id_ato: str, coluna_texto_entidade: str,
                 coluna_tipo_entidade: str, keep_punctuation: bool = False,
                 return_df: bool = False):
        self.coluna_id_ato = coluna_id_ato
        self.coluna_texto_entidade = coluna_texto_entidade
        self.coluna_tipo_entidade = coluna_tipo_entidade
        if not keep_punctuation:
            self.tokenizer = RegexpTokenizer('\w+')
        else:
            self.tokenizer = False
        self.return_df = return_df
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def gera_listas_atos_iobs(self, df):
        
        def _inclui_tags_vazias(texto_iob):
            texto_ato_iob = texto_iob.copy()
            for idx, token in enumerate(texto_ato_iob):
                if token[0:2] == 'B-':# in token:
                    pass
                elif token[0:2] == 'I-':# in token:
                    pass
                else:
                    texto_ato_iob[idx] = 'O'
            
            return texto_ato_iob
        
        def _constroi_iob(texto_ent, tipo_ent):
            if self.tokenizer:
                texto_ent_tok = self.tokenizer.tokenize(texto_ent)
            else:
                texto_ent_tok = word_tokenize(texto_ent)
            iob_entidade = []
            for index_token, token in enumerate(texto_ent_tok):
                # primeiro token?
                if index_token == 0:
                    palavra = 'B-'+ tipo_ent
                    iob_entidade.append(palavra)
                # é o segundo token?
                else:
                    palavra = 'I-'+ tipo_ent
                    iob_entidade.append(palavra)
            # salva tupla contendo texto tokenizado e iob correspondente
            if self.tokenizer:
                tup_entidade = (self.tokenizer.tokenize(texto_ent), iob_entidade)
            else:
                tup_entidade = (word_tokenize(texto_ent), iob_entidade)
            return tup_entidade
        
        def _match_iob_texto_ato(texto_entidade_tok, iob_ato):
            texto_ato_iob = texto_entidade_tok.copy()
            #print(iob_ato)
            for tupla in iob_ato:
                 # checa se o texto de referência existe
                if tupla[0]:
                    for i in range(len(texto_entidade_tok)):
                        # checa se a tag existe
                        if tupla[0][0]:
                            # match primeiro token
                            if texto_entidade_tok[i] == tupla[0][0]:
                                # a sequência de tokens de texto_entidade_token na
                                # posição encontrada é igual aos tokens da entidade?
                                if texto_entidade_tok[i:i+len(tupla[0])] == tupla[0]:
                                    texto_ato_iob[i:i+len(tupla[0])] = tupla[1]
            
            return texto_ato_iob

        atos = []
        lista_labels = []
        id_atos = set()
        for row in df.iterrows():
            id_ato = df.iloc[row[0]][self.coluna_id_ato]
            texto_ato = []
            texto_ato_iob = []
            if id_ato not in id_atos:
                id_atos.add(id_ato)
                lista_ids = list(df.query(f'{self.coluna_id_ato} == "{id_ato}"').index)
                # print(lista_ids)
                iob_ato = []
                # todas as anotações que não são o ato inteiro
                for index in lista_ids:
                    texto_entidade = df.iloc[index][self.coluna_texto_entidade]
                    tipo_entidade = df.iloc[index][self.coluna_tipo_entidade]
                    if isinstance(self.tokenizer, RegexpTokenizer):
                        texto_entidade_tok = self.tokenizer.tokenize(texto_entidade)
                    else:
                        texto_entidade_tok = word_tokenize(texto_entidade)
                    if not tipo_entidade.isupper():
                        tup_entidade = _constroi_iob(texto_entidade, tipo_entidade)
                        iob_ato.append(tup_entidade)
                # anotação do ato inteiro
                for index in lista_ids:
                    texto_entidade = df.iloc[index][self.coluna_texto_entidade]
                    tipo_entidade = df.iloc[index][self.coluna_tipo_entidade]
                    if self.tokenizer:
                        texto_entidade_tok = self.tokenizer.tokenize(texto_entidade)
                    else:
                        texto_entidade_tok = word_tokenize(texto_entidade)
                    if tipo_entidade.isupper():
                        texto_ato = texto_entidade_tok
                        texto_ato_iob = _match_iob_texto_ato(texto_entidade_tok, iob_ato)
                texto_ato_iob = _inclui_tags_vazias(texto_ato_iob)
                atos.append(texto_ato)
                lista_labels.append(texto_ato_iob)
        
        return atos, lista_labels

    def create_iob_df(self, atos, lista_labels):
        rows_list = []
        dict1 = {
                'Sentence_idx': -1,
                'Word': 'UNK',
                'Tag': 'O'
            }
        rows_list.append(dict1)
        id_ato = 0
        for ato, labels in zip(atos, lista_labels):
            for word, label in zip(ato, labels):
                dict1 = {
                    'Sentence_idx': id_ato,
                    'Word': word,
                    'Tag': label
                }
                rows_list.append(dict1)
                #print(word, label)
            id_ato += 1
        new_df = pd.DataFrame(rows_list)

        return new_df    
    
    def transform(self, df, **transform_params):
        dataframe = df.copy()
        dataframe = dataframe.reset_index(drop=True)
        atos, lista_labels = self.gera_listas_atos_iobs(dataframe)
        if self.return_df:
            iob_df = self.create_iob_df(atos, lista_labels)
            return iob_df
        else:
            return atos, lista_labels
        
        

