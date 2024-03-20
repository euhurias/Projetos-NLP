
import pandas as pd
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer

class IOBTransformer:
    def __init__(self, col_id_ato, col_texto_entidade, col_tipo_entidade, keep_punctuation=False, return_df=False):
        self.col_id_ato = col_id_ato
        self.col_texto_entidade = col_texto_entidade
        self.col_tipo_entidade = col_tipo_entidade
        self.keep_punctuation = keep_punctuation
        self.return_df = return_df
        self.tokenizer = RegexpTokenizer('\w+') if not keep_punctuation else None

    def fit(self, X, y=None, **fit_params):
        return self

    def _include_empty_tags(self, iob_tags):
        return ['O' if not tag.startswith(('B-', 'I-')) else tag for tag in iob_tags]

    def _build_iob_tags(self, entity_text, entity_type):
        tokens = self.tokenizer.tokenize(entity_text) if self.tokenizer else word_tokenize(entity_text)
        iob_tags = ['B-' + entity_type] + ['I-' + entity_type] * (len(tokens) - 1)
        return tokens, iob_tags

    def _match_iob_tags(self, entity_tokens, iob_acts):
        iob_tags = ['O'] * len(entity_tokens)
        for token_index, token in enumerate(entity_tokens):
            for iob_act_tokens, iob_act_tags in iob_acts:
                if token in iob_act_tokens:
                    start_index = iob_act_tokens.index(token)
                    if entity_tokens[token_index:token_index + len(iob_act_tokens[start_index:])] == iob_act_tokens[start_index:]:
                        iob_tags[token_index:token_index + len(iob_act_tokens)] = iob_act_tags[start_index:]
                        break
        return iob_tags

    def transform(self, df, **transform_params):
        atos = []
        lista_labels = []
        id_atos = set()
        for _, row in df.iterrows():
            id_ato = row[self.col_id_ato]
            if id_ato not in id_atos:
                id_atos.add(id_ato)
                ato_rows = df[df[self.col_id_ato] == id_ato]
                entity_rows = ato_rows[ato_rows[self.col_tipo_entidade].str.islower()]
                act_entities = [(row[self.col_texto_entidade], row[self.col_tipo_entidade]) for _, row in entity_rows.iterrows()]
                entity_tokens, entity_iobs = zip(*[self._build_iob_tags(text, entity_type) for text, entity_type in act_entities])
                iob_acts = [(tokens, tags) for tokens, tags in zip(entity_tokens, entity_iobs)]
                act_text = ato_rows.iloc[0][self.col_texto_entidade]
                act_tokens = self.tokenizer.tokenize(act_text) if self.tokenizer else word_tokenize(act_text)
                iob_tags = self._match_iob_tags(act_tokens, iob_acts)
                iob_tags = self._include_empty_tags(iob_tags)
                atos.append(act_tokens)
                lista_labels.append(iob_tags)

        if self.return_df:
            return self._create_iob_df(atos, lista_labels)
        else:
            return atos, lista_labels

    def _create_iob_df(self, atos, lista_labels):
        rows_list = [{'Sentence_idx': -1, 'Word': 'UNK', 'Tag': 'O'}]
        id_ato = 0
        for ato, labels in zip(atos, lista_labels):
            for word, label in zip(ato, labels):
                rows_list.append({'Sentence_idx': id_ato, 'Word': word, 'Tag': label})
            id_ato += 1
        return pd.DataFrame(rows_list)
