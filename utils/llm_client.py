"""
LLM 클라이언트 모듈
OpenAI, Anthropic, 로컬 모델 지원
"""
import os
from typing import Optional, Dict, Any
from config.settings import settings


class LLMClient:
    """LLM 클라이언트 래퍼 클래스"""
    
    def __init__(self):
        self.provider = getattr(settings, 'LLM_PROVIDER', 'openai').lower()
        self.model = getattr(settings, 'LLM_MODEL', 'gpt-3.5-turbo')
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """LLM 클라이언트 초기화"""
        try:
            if self.provider == 'openai':
                from openai import OpenAI
                api_key = os.getenv('OPENAI_API_KEY') or getattr(settings, 'OPENAI_API_KEY', None)
                if not api_key:
                    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
                self.client = OpenAI(api_key=api_key)
                
            elif self.provider == 'anthropic':
                from anthropic import Anthropic
                api_key = os.getenv('ANTHROPIC_API_KEY') or getattr(settings, 'ANTHROPIC_API_KEY', None)
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
                self.client = Anthropic(api_key=api_key)
                
            elif self.provider == 'local':
                # 로컬 모델 사용 (예: Ollama)
                try:
                    from openai import OpenAI
                    base_url = getattr(settings, 'LOCAL_LLM_BASE_URL', 'http://localhost:11434/v1')
                    api_key = 'ollama'  # Ollama는 키가 필요 없지만 일부 클라이언트는 요구함
                    self.client = OpenAI(base_url=base_url, api_key=api_key)
                except Exception as e:
                    print(f"로컬 LLM 연결 실패: {e}")
                    raise
            else:
                raise ValueError(f"지원하지 않는 LLM 제공자: {self.provider}")
                
            print(f"✓ LLM 클라이언트 초기화 완료 ({self.provider}/{self.model})")
            
        except ImportError as e:
            print(f"⚠️ LLM 라이브러리가 설치되지 않았습니다: {e}")
            print("pip install openai 또는 pip install anthropic 필요")
            self.client = None
        except Exception as e:
            print(f"⚠️ LLM 클라이언트 초기화 실패: {e}")
            self.client = None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.3, max_tokens: int = 2000) -> Optional[str]:
        """
        텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            temperature: 생성 온도 (0.0-1.0)
            max_tokens: 최대 토큰 수
            
        Returns:
            생성된 텍스트 또는 None (실패 시)
        """
        if not self.client:
            return None
        
        try:
            if self.provider == 'openai' or self.provider == 'local':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
                
            elif self.provider == 'anthropic':
                messages = []
                if system_prompt:
                    # Anthropic은 시스템 메시지를 별도 파라미터로 받음
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}]
                    )
                else:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                return response.content[0].text.strip()
                
        except Exception as e:
            print(f"❌ LLM 생성 실패: {e}")
            return None
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[Dict]:
        """
        JSON 형식으로 텍스트 생성
        
        Returns:
            파싱된 JSON 딕셔너리 또는 None
        """
        json_prompt = f"{prompt}\n\n반드시 JSON 형식으로만 응답하세요."
        if system_prompt:
            json_system = f"{system_prompt}\n응답은 반드시 유효한 JSON 형식이어야 합니다."
        else:
            json_system = "응답은 반드시 유효한 JSON 형식이어야 합니다."
        
        result = self.generate(json_prompt, json_system, temperature=0.1)
        
        if not result:
            return None
        
        try:
            import json
            # JSON 코드 블록 제거
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            return json.loads(result)
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패: {e}")
            print(f"원본 응답: {result}")
            return None


# 전역 LLM 클라이언트 인스턴스
_llm_client = None

def get_llm_client() -> Optional[LLMClient]:
    """전역 LLM 클라이언트 반환 (싱글톤)"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client if _llm_client.client is not None else None

