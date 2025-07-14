import os
import time
import logging
from typing import Optional, Dict, Any, Union, List
import httpx
import backoff

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cosyvoice-client")

# 常量定义
class ClientConfig:
    # 服务配置
    BASE_URL = "http://localhost:8000"  # 服务地址
    TIMEOUT = 60.0  # 请求超时时间(秒)
    
    # TTS 参数默认值
    DEFAULT_VOICE = "sft_中文女"  # 默认音色
    DEFAULT_SPEED = 1.0  # 默认语速
    DEFAULT_VOLUME = 1.0  # 默认音量
    DEFAULT_FORMAT = "wav"  # 默认音频格式
    
    # 重试配置
    MAX_RETRIES = 3  # 最大重试次数
    MAX_RETRY_TIME = 60  # 最大重试时间(秒)
    
    # 输出配置
    OUTPUT_DIR = "output"  # 输出目录


class CosyVoiceClient:
    """CosyVoice TTS API 客户端"""
    
    def __init__(
        self, 
        base_url: str = ClientConfig.BASE_URL,
        timeout: float = ClientConfig.TIMEOUT
    ):
        """初始化客户端
        
        Args:
            base_url: API 基础 URL
            timeout: 请求超时时间(秒)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        
        # 确保输出目录存在
        os.makedirs(ClientConfig.OUTPUT_DIR, exist_ok=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
    
    @backoff.on_exception(
        backoff.expo,
        (httpx.RequestError, httpx.HTTPStatusError),
        max_tries=ClientConfig.MAX_RETRIES,
        max_time=ClientConfig.MAX_RETRY_TIME,
        on_backoff=lambda details: logger.warning(
            f"重试请求 (尝试 {details['tries']}/{ClientConfig.MAX_RETRIES})"
        )
    )
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> httpx.Response:
        """发送 HTTP 请求并处理错误
        
        Args:
            method: HTTP 方法
            endpoint: API 端点
            **kwargs: 传递给 httpx 的其他参数
            
        Returns:
            httpx.Response: HTTP 响应
            
        Raises:
            httpx.RequestError: 请求错误
            httpx.HTTPStatusError: HTTP 状态错误
        """
        url = f"{self.base_url}{endpoint}"
        response = self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    
    def get_model_status(self) -> Dict[str, Any]:
        """获取模型状态
        
        Returns:
            Dict[str, Any]: 模型状态信息
        """
        response = self._request("GET", "/model/status")
        return response.json()
    
    def load_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """手动加载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 加载结果
        """
        params = {}
        if model_name:
            params["model"] = model_name
        
        response = self._request("POST", "/model/load", params=params)
        return response.json()
    
    def unload_model(self) -> Dict[str, Any]:
        """手动卸载模型
        
        Returns:
            Dict[str, Any]: 卸载结果
        """
        response = self._request("POST", "/model/unload")
        return response.json()
    
    def get_sft_voices(self) -> List[str]:
        """获取模型自带音色列表
        
        Returns:
            List[str]: 音色列表
        """
        response = self._request("POST", "/audio/voices_sft")
        return response.json()["voices"]
    
    def get_voices(self) -> List[str]:
        """获取预装音色列表
        
        Returns:
            List[str]: 音色列表
        """
        response = self._request("POST", "/audio/voices")
        return response.json()["voices"]
    
    def listen_voice(self, voice: str) -> bytes:
        """试听预装音色
        
        Args:
            voice: 音色名称
            
        Returns:
            bytes: 音频数据
        """
        response = self._request("GET", "/audio/voices/listen", params={"voice": voice})
        return response.content
    
    def generate_speech(
        self,
        text: str,
        voice: str = ClientConfig.DEFAULT_VOICE,
        speed: float = ClientConfig.DEFAULT_SPEED,
        volume_multiplier: float = ClientConfig.DEFAULT_VOLUME,
        response_format: str = ClientConfig.DEFAULT_FORMAT,
        stream: bool = False,
        save_path: Optional[str] = None
    ) -> Union[bytes, str]:
        """生成语音
        
        Args:
            text: 要转换为语音的文本
            voice: 音色，格式为 "模式_音色名称"
            speed: 语速，范围 0.25-4.0
            volume_multiplier: 音量倍数
            response_format: 音频格式，目前仅支持 wav
            stream: 是否使用流式传输
            save_path: 保存路径，如不指定则自动生成
            
        Returns:
            Union[bytes, str]: 如果 save_path 为 None，返回音频数据；否则返回保存路径
        """
        # 构建请求数据
        data = {
            "input": text,
            "voice": voice,
            "speed": speed,
            "volume_multiplier": volume_multiplier,
            "response_format": response_format,
            "stream": stream
        }
        
        # 发送请求
        logger.info(f"生成语音: '{text[:30]}{'...' if len(text) > 30 else ''}' (音色: {voice})")
        
        if stream:
            # 流式处理
            with self._request(
                "POST", 
                "/audio/speech", 
                json=data, 
                stream=True
            ) as response:
                # 如果需要保存
                if save_path is None:
                    save_path = self._generate_filename(text, voice, response_format)
                
                # 保存流式数据
                with open(save_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                
                logger.info(f"语音已保存至: {save_path}")
                return save_path
        else:
            # 非流式处理
            response = self._request("POST", "/audio/speech", json=data)
            audio_data = response.content
            
            # 如果需要保存
            if save_path is not None:
                with open(save_path, "wb") as f:
                    f.write(audio_data)
                logger.info(f"语音已保存至: {save_path}")
                return save_path
            
            return audio_data
    
    def _generate_filename(self, text: str, voice: str, format: str) -> str:
        """生成文件名
        
        Args:
            text: 文本
            voice: 音色
            format: 格式
            
        Returns:
            str: 文件路径
        """
        # 使用文本前10个字符作为文件名的一部分
        text_part = "".join(c for c in text[:10] if c.isalnum() or c.isspace()).strip()
        text_part = text_part.replace(" ", "_")
        
        # 使用时间戳确保唯一性
        timestamp = int(time.time())
        
        # 构建文件名
        filename = f"{voice}_{text_part}_{timestamp}.{format}"
        return os.path.join(ClientConfig.OUTPUT_DIR, filename)


# 使用示例
def main():
    # 创建客户端
    with CosyVoiceClient() as client:
        try:
            # 检查模型状态
            status = client.get_model_status()
            print(f"模型状态: {status}")
            
            # 如果模型未加载，则加载模型
            if not status.get("model_loaded", False):
                print("正在加载模型...")
                client.load_model()
            
            # 获取可用音色
            sft_voices = client.get_sft_voices()
            print(f"模型自带音色: {sft_voices}")
            
            # 生成语音示例
            text = "这是一个测试语音，使用CosyVoice进行文本转语音。"
            
            # 非流式生成并保存
            output_path = client.generate_speech(
                text=text,
                voice="sft_中文女",
                speed=1.0,
                volume_multiplier=1.0,
                stream=False
            )
            print(f"非流式生成完成，保存至: {output_path}")
            
            # 流式生成并保存
            output_path = client.generate_speech(
                text=text,
                voice="sft_中文男",
                speed=1.2,
                stream=True
            )
            print(f"流式生成完成，保存至: {output_path}")
            
        except Exception as e:
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
