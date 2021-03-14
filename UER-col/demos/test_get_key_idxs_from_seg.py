import torch
from samples.sample_segs import seg_new
from col_spec_yh.encode_utils import get_key_idxs

def test(seg):
    '''
    :param seg -> torch.LongTensor; shape: [bz, seq_len]

    A profile:
        [1]
        seg % 10000
            -> same number -> token in the same cell

        [2]
        seg // 10000
            ->
                0       -> [CLS (TABLE)] and [SEP (COLUMN)]
                n(>0)   -> the nth token in a cell

        [3]
        col_map: (seg % 10000) // 100

        [4]
        row_map: seg % 100
            99 -> [CLS]
            0 -> [SEP(COL)](s)
            n(>1) -> tokens in the nth row
    :return:
    '''
    assert seg.type() == 'torch.LongTensor'
    assert len(seg.shape) >= 2
    # import ipdb; ipdb.set_trace()
    get_key_idxs(seg)
    col_num = torch.max(seg%10000).item() // 100




    _t = torch.LongTensor([-10000]).repeat(seg.size()[0]).unsqueeze(-1).to(seg.device)
    _for_calc = torch.cat((_t, seg[:, :-1]), 1)
    _mid = seg - _for_calc
    import ipdb; ipdb.set_trace()
    _idxs = torch.nonzero((_mid > 0).float())
    import ipdb; ipdb.set_trace()



    print(seg)

if __name__=='__main__':
    test(seg_new)