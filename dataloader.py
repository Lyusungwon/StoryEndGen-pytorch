from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

# NOTE original repo used pattern.en.lemma, but since Pattern is incompatible with Python 3.7
# (Refer to https://github.com/clips/pattern/issues/283)
# We replaced it with WordNetLemmatizer.

# Configurations
# TODO change config variables below to flags
_START_VOCAB = ['_PAD', '_UNK', '_SOS', '_EOS', '_NAF_H', '_NAF_R', '_NAF_T']
vocab_size = 10000  # vocab size
triple_num = 10  # max num of triples for each head entity

lemmatizer = WordNetLemmatizer()


def get_dataloader(data_path='data',
                   data_name='train',
                   batch_size=128,
                   shuffle=True,
                   num_workers=4):
    global relation, transform, dataset

    dataset = ROCStoriesDataset(data_path, data_name)

    relation = dataset.rel_dict
    transform = dataset.transform
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=True,
                             collate_fn=collate_text
                             )

    return data_loader


class ROCStoriesDataset(Dataset):
    def __init__(self, data_path='data', data_name='train'):
        assert data_name in ['train', 'test', 'val'], "Data name should be among ['train', 'test', 'val']."

        self.data_path = data_path
        self.data_name = data_name

        self.data = self.load_data(data_name)
        # self.data: [{'post': [[sent1], [sent2], [sent3], [sent4]], 'response': [sent]}, ...]
        self.vocab_list, self._vocab_dict = self.build_vocab()
        self.rel_dict = self.load_relation()

        self.idx2word = OrderedDict([(idx, vocab) for idx, vocab in enumerate(self.vocab_list)])
        self.word2idx = OrderedDict([(vocab, idx) for idx, vocab in enumerate(self.vocab_list)])

    def load_data(self, data_name):
        post = []
        with open(f'{self.data_path}/{data_name}.post', 'r', encoding='latin-1') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split("\t")
                post.append([p.split() for p in tmp])

        with open(f'{self.data_path}/{data_name}.response', 'r', encoding='latin-1') as f:
            response = [line.strip().split() for line in f.readlines()]

        data = []
        for p, r in zip(post, response):
            data.append({'post': p, 'response': r})

        return data

    def build_vocab(self):
        relation_vocab_list = []
        relation_file = open(f'{self.data_path}/relations.txt', 'r')
        for line in relation_file:
            relation_vocab_list += line.strip().split()

        vocab_dict = {}
        for i, pair in enumerate(self.load_data('train')):  # NOTE only use train set vocab
            for token in [word for sent in pair['post'] for word in sent] + pair['response']:
                if token in vocab_dict:
                    vocab_dict[token] += 1
                else:
                    vocab_dict[token] = 1
        vocab_list = _START_VOCAB + relation_vocab_list + sorted(vocab_dict, key=vocab_dict.get, reverse=True)
        # tokens in vocab_dict are sorted by their term frequency in corpus

        if len(vocab_list) > vocab_size:
            vocab_list = vocab_list[:vocab_size]  # limit vocab_size

        return vocab_list, vocab_dict

    def load_relation(self):
        """
        :return: rel_dict['hi'] = [['hi', '/r/RelatedTo', 'friendly'], ['hi', '/r/RelatedTo', 'high'], ...]
                 each head entity has `triple_num` rel_dict triples, sorted by tail entity frequency in corpus
        """
        file = open(f'{self.data_path}/triples_shrink.txt', "r")

        rel_dict = {}
        for line in file:
            tmp = line.strip().split()
            if tmp[0] in rel_dict:
                if tmp[2] not in rel_dict[tmp[0]]:
                    rel_dict[tmp[0]].append(tmp)
            else:
                rel_dict[tmp[0]] = [tmp]

        for r in rel_dict.keys():  # r: head entity of each rel_dict
            tmp_vocab = {}
            i = 0
            for re in rel_dict[r]:
                if re[2] in self._vocab_dict.keys():  # if tail entity is in vocab_dict,
                    tmp_vocab[i] = self._vocab_dict[re[2]]  # tmp_vocab[tail entity index] = tail entity 빈도수
                i += 1
            tmp_list = sorted(tmp_vocab, key=tmp_vocab.get)[:triple_num] if len(
                tmp_vocab) > triple_num else sorted(tmp_vocab, key=tmp_vocab.get)

            new_relation = []
            for i in tmp_list:
                new_relation.append(rel_dict[r][i])
            rel_dict[r] = new_relation

        return rel_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        posts = self.data[i]['post']
        response = self.data[i]['response']

        return posts, response

    def transform(self, word):
        """ Converts word to idx in vocabulary """
        if word in self.vocab_list:
            return self.word2idx[word]
        else:
            return 1


def collate_text(batch):
    batch_posts, batch_response = list(zip(*batch))

    max_post_len_list = [max([len(posts[i]) for posts in batch_posts]) + 2 for i in range(4)]
    max_response_len = max([len(response) for response in batch_response]) + 2

    post_1, post_2, post_3, post_4 = [], [], [], []
    post_length_1, post_length_2, post_length_3, post_length_4 = [], [], [], []
    responses = []
    responses_length = []

    def padding(sent, length):
        """ Add sos and eos tokens, then pad sentence to length"""
        return ['_SOS'] + sent + ['_EOS'] + (['_PAD'] * (length - len(sent) - 2))

    for posts in batch_posts:
        post_1.append(padding(posts[0], max_post_len_list[0]))
        post_2.append(padding(posts[1], max_post_len_list[1]))
        post_3.append(padding(posts[2], max_post_len_list[2]))
        post_4.append(padding(posts[3], max_post_len_list[3]))

        post_1[-1] = list(map(transform, post_1[-1]))
        post_2[-1] = list(map(transform, post_2[-1]))
        post_3[-1] = list(map(transform, post_3[-1]))
        post_4[-1] = list(map(transform, post_4[-1]))

        post_length_1.append(len(posts[0]) + 2)
        post_length_2.append(len(posts[1]) + 2)
        post_length_3.append(len(posts[2]) + 2)
        post_length_4.append(len(posts[3]) + 2)

    for i in range(len(batch_response)):
        response = batch_response[i]
        responses.append(padding(response, max_response_len))
        responses[-1] = list(map(transform, responses[-1]))
        responses_length.append(len(response) + 2)

    entity = [[], [], [], []]
    for posts in batch_posts:
        for i in range(4):
            entity[i].append([])
            for j in range(len(posts[i])):
                word = posts[i][j]
                try:
                    lemmatized = lemmatizer.lemmatize(word)
                except UnicodeEncodeError:
                    lemmatized = word
                if lemmatized in relation:
                    entity[i][-1].append([list(map(transform, triple)) for triple in relation[lemmatized]])
                else:
                    entity[i][-1].append([[4, 5, 6]])  # naf_h, naf_r, naf_t

    # entity[i][j][k][l] : lth triple with kth word in ith post of jth sample as head entity

    max_triple_len = [0, 0, 0, 0]
    for i in range(4):
        for item in entity[i]:
            for triple in item:
                if len(triple) > max_triple_len[i]:
                    max_triple_len[i] = len(triple)

    entity_length_list = []
    for i in range(4):
        entity_length_list.append(torch.zeros(len(batch_posts), max_post_len_list[i]))

    for i in range(4):
        for j in range(len(entity[i])):
            entity[i][j] = [[[4, 5, 6]]] + entity[i][j] + [[[4, 5, 6]]]  # add padding for 'sos', 'eos'
            for k in range(len(entity[i][j])):
                entity_length_list[i][j][k] = len(entity[i][j][k])

                # pad each triple list to max_triple_length
                if len(entity[i][j][k]) < max_triple_len[i]:
                    entity[i][j][k] = entity[i][j][k] + [[4, 5, 6]] * (max_triple_len[i] - len(entity[i][j][k]))

            # pad each post_i to (max_post_len_list, max_triple_len)
            if len(entity[i][j]) < (max_post_len_list[i]):
                entity[i][j] = entity[i][j] + [[[4, 5, 6]] * max_triple_len[i]] * (max_post_len_list[i] - len(entity[i][j]))

    entity_1 = torch.tensor(entity[0])
    entity_mask_1 = entity_1[:, :, :, 0] == 4
    entity_2 = torch.tensor(entity[1])
    entity_mask_2 = entity_2[:, :, :, 0] == 4
    entity_3 = torch.tensor(entity[2])
    entity_mask_3 = entity_3[:, :, :, 0] == 4
    entity_4 = torch.tensor(entity[3])
    entity_mask_4 = entity_4[:, :, :, 0] == 4

    batched_data = {
        'post_1': torch.tensor(post_1),  # (batch_size, max_post_1_len)
        'post_2': torch.tensor(post_2),
        'post_3': torch.tensor(post_3),
        'post_4': torch.tensor(post_4),
        'post_length_1': torch.tensor(post_length_1),  # (batch_size,)
        'post_length_2': torch.tensor(post_length_2),
        'post_length_3': torch.tensor(post_length_3),
        'post_length_4': torch.tensor(post_length_4),
        'responses': torch.tensor(responses),  # (batch_size, max_response_len)
        'responses_length': torch.tensor(responses_length),  # (batch_size,)
        'entity_1': entity_1,  # (batch_size, max_post_1_len, max_triple_num, 3)
        'entity_2': entity_2,
        'entity_3': entity_3,
        'entity_4': entity_4,
        'entity_mask_1': entity_mask_1,  # (batch_size, max_post_1_len, max_triple_num)
        'entity_mask_2': entity_mask_2,
        'entity_mask_3': entity_mask_3,
        'entity_mask_4': entity_mask_4,
        'entity_length_1': entity_length_list[0],  # (batch_size, max_post_1_len)
        'entity_length_2': entity_length_list[1],
        'entity_length_3': entity_length_list[2],
        'entity_length_4': entity_length_list[3]
    }

    return batched_data


if __name__ == "__main__":
    dataloader = get_dataloader(batch_size=3, shuffle=False)
    for batch in dataloader:
        print("Size of batch properties with bach_size 3\n")

        print('post_1:', batch['post_1'].size())
        print('post_length_1:', batch['post_length_1'].size())
        print('responses:', batch['responses'].size())
        print('responses_length:', batch['responses_length'].size())
        print('entity_1:', batch['entity_1'].size())
        print('entity_mask_1:', batch['entity_mask_1'].size())
        print('entity_length_1:', batch['entity_length_1'].size())

        break

