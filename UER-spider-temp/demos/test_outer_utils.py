from torch_scatter import scatter_mean, scatter_max, scatter_add
import torch

def test_1():
    index = torch.tensor([0, 0, 1, 0, 2, 2, 3, 3])
    input = torch.tensor([5, 1, 7, 2, 3, 2, 1, 3])
    _ = scatter_add(input, index)
    assert torch.equal(_, torch.LongTensor([8, 7, 5, 4]))  # footnote: torch.eq vs torch.equal


def test_2():
    index = torch.tensor([1, 0, 2, 3])
    input = torch.tensor([0, 1, 2, 3])
    _ = scatter_add(input, index)
    assert torch.equal(_, torch.LongTensor([1, 0, 2, 3]))  # footnote: torch.eq vs torch.equal


def test_3():
    input = torch.tensor([0, 1, 2, 3])
    index = torch.tensor([0, 1, 2, 4])
    _ = scatter_add(input, index)
    assert torch.equal(_, torch.LongTensor([0, 1, 2, 0, 3]))  # footnote: torch.eq vs torch.equal


def test_4():
    src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
    index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
    out = scatter_add(src, index, dim=-1)
    assert torch.equal(out, torch.LongTensor([[0, 0, 4, 3, 3, 0], [2, 4, 4, 0, 0, 0]]))
    max, argmax = scatter_max(src, index, dim=-1)
    import ipdb; ipdb.set_trace()
    assert torch.equal(max, torch.LongTensor([[0, 0, 4, 3, 2, 0], [2, 4, 3, 0, 0, 0]]))  # not exists -> 0 (?)
    assert torch.equal(argmax, torch.LongTensor([[5, 5, 3, 4, 0, 1], [1, 4, 3, 5, 5, 5]]))  # out of bound -> 5
    # import ipdb; ipdb.set_trace()


def test_4_1():
    src = torch.tensor([[2, -1, 1, 4, 3], [-1, 2, 1, 3, 4]])
    index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
    out = scatter_add(src, index, dim=-1)
    # assert torch.equal(out, torch.LongTensor([[0, 0, 4, 3, 3, 0], [2, 4, 4, 0, 0, 0]]))
    import ipdb; ipdb.set_trace()
    max, argmax = scatter_max(src, index, dim=-1)
    # assert torch.equal(max, torch.LongTensor([[0, 0, 4, 3, 2, 0], [2, 4, 3, 0, 0, 0]]))  # not exists -> 0 (?)
    # assert torch.equal(argmax, torch.LongTensor([[5, 5, 3, 4, 0, 1], [1, 4, 3, 5, 5, 5]]))  # out of bound -> 5


if __name__=='__main__':
    test_4()