import pytest

from hisim.dataset.base_dataset import GenericRequest, SimpleDataset
from hisim.dataset.dataset_args import DatasetArgs
from hisim.dataset.prefix_cache import PrefixCacheDecorator


class RecordingDataset(SimpleDataset):
    def __init__(self, prefix_hit_rate=0.5):
        requests = [
            GenericRequest(token_ids=[token] * 4, input_length=4, output_length=1)
            for token in range(1, 5)
        ]
        super().__init__(args=DatasetArgs(prefix_hit_rate=prefix_hit_rate), reqs=requests)
        self.reads = []

    def __getitem__(self, index):
        self.reads.append(index)
        return super().__getitem__(index)


def test_prefix_dataset_supports_out_of_order_and_repeated_access():
    source = RecordingDataset()
    dataset = PrefixCacheDecorator(source)

    assert dataset[2].token_ids == [1, 1, 3, 3]
    assert source.reads == [0, 1, 2]
    assert dataset[0].token_ids == [1, 1, 1, 1]
    assert dataset[2].token_ids == [1, 1, 3, 3]
    assert source.reads == [0, 1, 2]

    assert dataset[-1].token_ids == [1, 1, 4, 4]
    assert source.reads == [0, 1, 2, 3]
    with pytest.raises(IndexError):
        dataset[4]
    with pytest.raises(IndexError):
        dataset[-5]


def test_prefix_dataset_supports_slices_with_nonzero_start_and_reverse_order():
    source = RecordingDataset()
    dataset = PrefixCacheDecorator(source)

    assert [req.token_ids for req in dataset[1:3]] == [[1, 1, 2, 2], [1, 1, 3, 3]]
    assert [req.token_ids for req in dataset[::-1]] == [
        [1, 1, 4, 4], [1, 1, 3, 3], [1, 1, 2, 2], [1, 1, 1, 1]
    ]
    assert source.reads == [0, 1, 2, 3]


def test_prefix_dataset_without_prefix_hits_delegates():
    source = RecordingDataset(prefix_hit_rate=None)
    dataset = PrefixCacheDecorator(source)

    assert dataset[2] is source.data[2]
    assert dataset[2] is source.data[2]
    assert source.reads == [2, 2]


def test_prefix_dataset_preserves_order_for_prompt_strings():
    class Tokenizer:
        def encode(self, prompt, add_special_tokens=False):
            return list(map(ord, prompt))

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(map(chr, token_ids))

    requests = [
        GenericRequest(prompt=prompt, input_length=4, output_length=1)
        for prompt in ("abcd", "wxyz", "1234")
    ]
    source = SimpleDataset(
        tokenizer=Tokenizer(), args=DatasetArgs(prefix_hit_rate=0.5), reqs=requests
    )
    dataset = PrefixCacheDecorator(source)

    assert [req.prompt for req in dataset] == ["abcd", "abwx", "ab12"]
    assert dataset[1].prompt == "abwx"
