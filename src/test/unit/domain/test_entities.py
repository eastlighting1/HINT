import torch
import torch.nn as nn
from unittest.mock import MagicMock
from hint.domain.entities import TrainableEntity
from hint.foundation.configs import TrainingConfig
from hint.foundation.dtos import TensorBatch

def test_trainable_entity_lifecycle():
    # Setup
    network = nn.Linear(10, 2)
    config = TrainingConfig(lr=0.1, epochs=1)
    entity = TrainableEntity(network, config)
    
    # Check initialization
    assert entity.state == "INITIALIZED"
    assert entity.epoch == 0
    
    # Mock Batch
    batch = TensorBatch(
        x_num=torch.randn(4, 10), # Simplified for Linear
        x_cat=None,
        targets=torch.tensor([0, 1, 0, 1]),
        stay_ids=[1, 2, 3, 4]
    )
    
    # Mock Loss Fn
    loss_fn = nn.CrossEntropyLoss()
    
    # Mock Network behavior for entity wrapper test (bypass complex forward)
    # Here we actually use real Linear layer to test optimizer step
    initial_weight = network.weight.data.clone()
    
    # Action
    loss = entity.step_train(batch, loss_fn)
    
    # Assert
    assert loss > 0
    assert not torch.equal(network.weight.data, initial_weight) # Weights updated
    
def test_entity_snapshot_restore():
    network = nn.Linear(10, 2)
    config = TrainingConfig()
    entity = TrainableEntity(network, config)
    entity.epoch = 5
    entity.best_metric = 0.85
    
    snapshot = entity.snapshot()
    
    new_entity = TrainableEntity(nn.Linear(10, 2), config)
    new_entity.restore(snapshot)
    
    assert new_entity.epoch == 5
    assert new_entity.best_metric == 0.85
    assert new_entity.id == entity.id