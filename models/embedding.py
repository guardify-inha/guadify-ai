"""
다중 임베딩 모델 시스템

이 모듈은 여러 SentenceTransformer 모델을 관리하고 임베딩을 생성하는 
MultiEmbeddingModel 클래스를 제공합니다.
"""

from typing import List, Dict, Union, Optional
import numpy as np
import logging
from sentence_transformers import SentenceTransformer


class MultiEmbeddingModel:
    """
    여러 SentenceTransformer 모델을 관리하고 임베딩을 생성하는 클래스
    
    이 클래스는 다음 세 가지 모델을 지원합니다:
    - all-MiniLM-L6-v2: 일반적인 영어 텍스트에 최적화
    - paraphrase-multilingual-MiniLM-L12-v2: 다국어 지원
    - jhgan/ko-sbert-nli: 한국어 특화 모델
    """
    
    # 지원하는 모델 목록
    SUPPORTED_MODELS = [
        "all-MiniLM-L6-v2",
        "paraphrase-multilingual-MiniLM-L12-v2", 
        "jhgan/ko-sbert-nli"
    ]
    
    def __init__(self, model_names: Optional[List[str]] = None, device: Optional[str] = None):
        """
        MultiEmbeddingModel 인스턴스를 초기화합니다.
        
        Args:
            model_names (List[str], optional): 사용할 모델 이름 목록.
                                              None인 경우 모든 지원 모델을 사용.
            device (str, optional): 모델을 로드할 디바이스 ('cpu', 'cuda', 'mps' 등).
        """
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 모델 이름 설정
        if model_names is None:
            model_names = self.SUPPORTED_MODELS
            
        self.model_names = model_names
        self.models = {}
        
        # 모델 로딩
        self._load_models(device)
    
    def _load_models(self, device: Optional[str] = None):
        """
        지정된 모델들을 로드합니다.
        
        Args:
            device (str, optional): 모델을 로드할 디바이스
        """
        self.logger.info(f"로딩할 모델 목록: {self.model_names}")
        
        for model_name in self.model_names:
            if model_name not in self.SUPPORTED_MODELS:
                self.logger.warning(f"지원하지 않는 모델 '{model_name}'이 건너뛰어집니다.")
                continue
                
            try:
                self.logger.info(f"모델 '{model_name}' 로딩 시작...")
                self.models[model_name] = SentenceTransformer(model_name, device=device)
                self.logger.info(f"모델 '{model_name}' 로딩 완료")
                
            except Exception as e:
                self.logger.error(f"모델 '{model_name}' 로딩 실패: {str(e)}")
                raise RuntimeError(f"모델 '{model_name}' 로딩 중 오류 발생: {str(e)}")
        
        self.logger.info(f"총 {len(self.models)}개 모델이 성공적으로 로드되었습니다.")
    
    def encode(self, texts: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        입력된 텍스트에 대해 모든 모델의 임베딩을 생성합니다.
        
        Args:
            texts (Union[str, List[str]]): 임베딩을 생성할 텍스트 또는 텍스트 리스트
            
        Returns:
            Dict[str, np.ndarray]: 모델 이름을 키로, 임베딩을 값으로 하는 딕셔너리
            
        Raises:
            NotImplementedError: 아직 구현되지 않음
        """
        # TODO: 실제 임베딩 생성 로직은 다음 서브태스크에서 구현
        raise NotImplementedError("encode 메서드는 아직 구현되지 않았습니다.")
    
    def get_model_info(self) -> Dict[str, Dict]:
        """
        로드된 모델들의 정보를 반환합니다.
        
        Returns:
            Dict[str, Dict]: 모델 정보 딕셔너리
        """
        model_info = {}
        for model_name, model in self.models.items():
            model_info[model_name] = {
                "model_name": model_name,
                "max_seq_length": getattr(model, 'max_seq_length', None),
                "device": str(model.device) if hasattr(model, 'device') else None,
                "is_loaded": True
            }
        return model_info
    
    def __repr__(self) -> str:
        """클래스의 문자열 표현을 반환합니다."""
        return f"MultiEmbeddingModel(models={list(self.models.keys())})"
