from omegaconf import DictConfig, OmegaConf
from hint.foundation.dtos import AppContext
from hint.domain.vo import ETLConfig, ICDConfig, CNNConfig

def load_app_context(cfg: DictConfig) -> AppContext:
    """
    Convert Hydra DictConfig to strongly typed AppContext.
    """
    # 1. ETL Config
    # ETL은 구조가 복잡해서(nested) 기존 방식 유지 혹은 필요한 부분만 언패킹
    etl_raw = cfg.get("etl", {}) if "etl" in cfg else cfg # 구조에 따라 조정
    # (기존 코드 유지하되, 만약 etl_config.yaml 구조와 Pydantic이 같다면 언패킹 가능)
    # 안전을 위해 기존 로직 유지:
    etl_cfg = ETLConfig(
        raw_dir=cfg.get("data", {}).get("raw_dir", "./data/raw"),
        proc_dir=cfg.get("data", {}).get("proc_dir", "./data/processed"),
        resources_dir=cfg.get("data", {}).get("resources_dir", "./resources"),
        # 필요하다면 아래처럼 추가 매핑 필요, 하지만 ICD가 급하므로 패스
    )
    
    # 2. ICD Config (여기가 문제였음)
    # Hydra Config 객체를 일반 Dict로 변환 (Safe)
    icd_raw = OmegaConf.to_container(cfg.get("icd", {}), resolve=True)
    
    # [핵심 수정] 일일이 나열하지 말고 **(언패킹)으로 한 번에 주입합니다.
    # 이렇게 하면 xai_sample_size, focal_gamma 등 YAML에 있는 모든 키가 자동으로 들어갑니다.
    icd_cfg = ICDConfig(**icd_raw)

    # 3. CNN Config
    cnn_raw = cfg.get("cnn", {})
    # CNN은 구조가 data/model로 나뉘어 있어서 단순 언패킹이 어렵습니다.
    # 하지만 여기도 빠진 게 없는지 확인해야 합니다.
    cnn_data = cnn_raw.get("data", {})
    cnn_model = cnn_raw.get("model", {})
    
    cnn_cfg = CNNConfig(
        data_path=cnn_data.get("path", "data/processed/dataset_123_inferred.parquet"),
        data_cache_dir=cnn_data.get("data_cache_dir", "data/cache"),
        exclude_cols=cnn_data.get("exclude_cols", ["ICD9_CODES"]),
        # model 섹션은 키가 많으므로 언패킹 추천
        **cnn_model 
    )

    return AppContext(
        etl=etl_cfg,
        icd=icd_cfg, # 이제 정상적으로 5가 들어간 객체가 리턴됩니다.
        cnn=cnn_cfg,
        mode=cfg.get("mode", "train"),
        seed=cfg.get("seed", 42)
    )