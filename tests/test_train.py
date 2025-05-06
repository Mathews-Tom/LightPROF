import pytest

from lightprof.train import DummyKGDataset

def test_dummy_kg_dataset_instantiation():
    dataset = DummyKGDataset(num_samples=1)
    assert isinstance(dataset, DummyKGDataset)