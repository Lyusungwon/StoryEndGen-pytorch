from torch.utils.data import Dataset, DataLoader
from pathlib import Path

home = str(Path.home())


def collate_text(list_inputs):
    pass


def load_dataloader(data_dir, batch_size=128, is_train=True, cpu_num=4):
    dataloader = DataLoader(
        StoryEndingGeneration(
            data_dir=data_dir,
            is_train=is_train,
            ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_num,
        pin_memory=True,
        collate_fn=collate_text)
    return dataloader


class StoryEndingGeneration(Dataset):
    def __init__(self, data_dir, is_train):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.load_data()

    def load_data(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    data_dir = 'data'
    dataloader = load_dataloader(data_dir, 128, True, 0)
    for batch in dataloader:
        print(batch['posts_1'].size()) # batch_size * variable length / Sorted in descending order
        print(batch['posts_length_1'].size()) # batch_size
        print(batch['entity_1'].size()) # batch_size * variable length * 10 * 3(tokens for h/r/t); one query(head) per timestep
        print(batch['entity_length_1'].size()) # batch_size
        print(batch['response'].size()) # batch_size * variable length
        
#         'posts_1': np.array(posts_1), # batch_size, sentence length, / sorted in descending order
#                     'posts_2': np.array(posts_2), 
#                     'posts_3': np.array(posts_3), 
#                     'posts_4': np.array(posts_4),       
#                     'entity_1': np.array(entity_0),
#                     'entity_2': np.array(entity_1),
#                     'entity_3': np.array(entity_2),
#                     'entity_4': np.array(entity_3),
#                     'entity_mask_1': np.array(entity_mask_0),
#                     'entity_mask_2': np.array(entity_mask_1),
#                     'entity_mask_3': np.array(entity_mask_2),
#                     'entity_mask_4': np.array(entity_mask_3),
#                      posts, entities sorted indices
#                     'posts_length_1': posts_length_1, 
#                     'posts_length_2': posts_length_2, 
#                     'posts_length_3': posts_length_3, 
#                     'posts_length_4': posts_length_4, 
#                     'responses': np.array(responses),
#                     'responses_length': responses_length
        break
