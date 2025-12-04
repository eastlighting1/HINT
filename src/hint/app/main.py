import hydra
from omegaconf import DictConfig
from .factory import AppFactory

@hydra.main(config_path="../../configs", config_name="cnn_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the HINT application.

    Orchestrates the creation of the Factory, Domain Objects, and Services,
    then executes the training workflow.

    Args:
        cfg: Hydra configuration object.
    """
    factory = AppFactory(cfg)
    
    hint_config = factory.create_configs()
    entity = factory.create_entity(hint_config)
    trainer = factory.create_service(hint_config)
    train_source, val_source = factory.create_sources(hint_config)
    
    trainer.train_model(
        entity=entity,
        train_source=train_source,
        val_source=val_source,
        epochs=hint_config.train.epochs
    )

if __name__ == "__main__":
    main()