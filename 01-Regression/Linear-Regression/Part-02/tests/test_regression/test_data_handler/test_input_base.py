import pytest
from models.data_handler import InputBase


@pytest.fixture(scope="module")
def input_base():
    input_base = InputBase()
    print(input_base)
    return input_base

def test_input_base(input_base):
    pass