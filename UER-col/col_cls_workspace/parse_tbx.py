import requests

# url = 'http://localhost:6006/data/plugin/scalars/scalars?tag=data%2Facc&'+\
#       'run='+'shuffle%2Bavg-cell-seg%2Bpositional-embedding-0-2-202101261300'+\
#       '%2Fdata%2Facc%2Ft2d&experiment=&format=csv'

a = 'http://localhost:6006/data/plugin/scalars/scalars?tag=data%2Facc&run='
b = '%2Fdata%2Facc%2F'
c = '&experiment=&format=csv'

# a _ b  [t2d/limaye/wikipedia] c
import os
path = './runs2/'

def parse_tbx_to_csv(out_name = 'acc_rec'):
    urls = [a+'%2B'.join(p.split('+'))+b for p in os.listdir(path)]
    tem = requests.get(urls[1]).content

    # exp_set; exp_full_name; epoch_num; ds_name; acc

    for p in os.listdir(path):
        for ds_name in ['t2d', 'limaye', 'wikipedia']:
            exp_set = '-'.join(p.split('-')[:-3])
            exp_full_name = p

            url = a+'%2B'.join(p.split('+')) + b + ds_name + c
            _ = requests.get(url).content.decode().strip()
            for _ in _.split('\r\n')[1:]:
                epoch_num = _.split(',')[1]
                acc = _.split(',')[2]
                print(exp_set, epoch_num, ds_name, acc, exp_full_name, file=open(out_name, 'a'))


# exp_set; exp_full_name; epoch_num; ds_name; acc
def parse_csv(path='acc_rec'):
    import pandas as pd
    df = pd.read_csv(path, sep=' ', header=None, names=['exp_set','epoch_num', 'ds_name',  'acc', 'exp_full_name'])
    import ipdb; ipdb.set_trace()
    exp_set_choices = df['exp_set'].drop_duplicates().tolist()
    epoch_num_choices = df['epoch_num'].drop_duplicates().tolist()
    ds_name_choices = df['ds_name'].drop_duplicates().tolist()




if __name__=='__main__':
    # parse_tbx_to_csv('acc_rec')
    parse_csv('acc_rec')