import logging
import os

import gradio as gr
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from webui.config import Config
from webui.router_model import engine, global_engine_lock
from webui.router_openai import voice_list, voice_sft_list

logger = logging.getLogger()

router = APIRouter(
    tags=["Web"],
    responses={404: {"description": "Not found"}},
)

templates = Jinja2Templates(directory="webui/templates")


@router.get("/", response_class=HTMLResponse)
async def get_web_ui(request: Request):
    """获取Web用户界面"""
    return templates.TemplateResponse("index.html", {"request": request})


def create_gradio_ui():
    """创建Gradio用户界面"""
    with gr.Blocks(title="CosyVoice TTS") as app:
        gr.Markdown("# CosyVoice 语音合成系统")
        
        with gr.Row():
            with gr.Column():
                # 创建模式选择下拉框
                mode = gr.Radio(
                    ["sft", "reference"], 
                    label="模式选择", 
                    value="sft",
                    info="选择SFT模式使用内置音色，选择Reference模式使用参考音色"
                )
                
                # 音色选择区域（动态更新）
                voice_dropdown = gr.Dropdown(
                    [], 
                    label="音色选择",
                    info="根据模式选择不同的音色"
                )
                
                # 试听按钮（仅在reference模式下显示）
                listen_button = gr.Button("试听音色", visible=False)
                
                # 文本输入区
                text_input = gr.Textbox(
                    label="输入文本", 
                    placeholder="请输入要转换成语音的文本...",
                    lines=5
                )
                
                # 速度调节
                speed_slider = gr.Slider(
                    minimum=0.5, 
                    maximum=2.0, 
                    value=1.0, 
                    step=0.05, 
                    label="语速",
                    info="调整语音播放速度"
                )
                
                # 音量调节
                volume_slider = gr.Slider(
                    minimum=0.5, 
                    maximum=2.0, 
                    value=1.0, 
                    step=0.05, 
                    label="音量",
                    info="调整语音音量"
                )
                
                # 生成按钮
                generate_button = gr.Button("生成语音", variant="primary")
            
            with gr.Column():
                # 音频输出区
                audio_output = gr.Audio(label="生成的语音", type="numpy")
                
                # 状态信息
                status_info = gr.Markdown("准备就绪，请输入文本并选择音色")
        
        # 更新音色列表
        def update_voice_list(mode_value):
            if mode_value == "sft":
                # 获取SFT模式下的音色列表
                response = voice_sft_list()
                voices = response.get("voices", [])
                return {"choices": voices, "visible": False}
            else:
                # 获取Reference模式下的音色列表
                response = voice_list()
                voices = response.get("voices", [])
                return {"choices": voices, "visible": True}
        
        # 生成语音功能
        def generate_speech(mode_value, voice_value, text, speed, volume):
            if not text.strip():
                return None, "错误：文本不能为空"
            
            if not voice_value:
                return None, "错误：请选择一个音色"
                
            try:
                with global_engine_lock:
                    voice_name = f"{mode_value}_{voice_value}"
                    
                    # 构建参数
                    prompt_wav_path = ""
                    prompt_wav_text = ""
                    
                    if mode_value != "sft":
                        prompt_wav_path = os.path.join(Config.voice_dir, voice_value)
                        prompt_wav_text_path = os.path.join(Config.voice_dir, voice_value + ".txt")
                        
                        if os.path.exists(prompt_wav_text_path):
                            with open(prompt_wav_text_path, "r") as f:
                                prompt_wav_text = f.read()
                    
                    # 生成音频
                    generator = engine.generate(
                        mode=mode_value,
                        spk_id=voice_value,
                        tts_text=text,
                        prompt_wav_path=prompt_wav_path,
                        prompt_wav_text=prompt_wav_text,
                        speed=speed,
                        stream=False,
                        max_val=volume
                    )
                    
                    # 收集音频片段
                    audio_chunks = []
                    for _, audio_data in generator:
                        audio_chunks.append(audio_data)
                    
                    if not audio_chunks:
                        return None, "错误：生成音频失败，没有返回任何数据"
                    
                    import numpy as np
                    combined_audio = np.concatenate(audio_chunks)
                    return (22050, combined_audio), "语音生成成功！"
            
            except Exception as e:
                logger.error(f"生成音频失败: {str(e)}")
                return None, f"错误：生成音频失败 - {str(e)}"
        
        # 试听音色功能
        def listen_voice(voice_value):
            if not voice_value:
                return None, "错误：请先选择一个音色"
            
            try:
                voice_path = os.path.join(Config.voice_dir, voice_value)
                if not os.path.exists(voice_path):
                    return None, f"错误：音色文件不存在 - {voice_value}"
                
                import soundfile as sf
                data, samplerate = sf.read(voice_path)
                return (samplerate, data), "试听音色"
            
            except Exception as e:
                logger.error(f"试听音色失败: {str(e)}")
                return None, f"错误：试听音色失败 - {str(e)}"
        
        # 事件绑定
        mode.change(
            fn=update_voice_list,
            inputs=[mode],
            outputs=[voice_dropdown, listen_button]
        )
        
        generate_button.click(
            fn=generate_speech,
            inputs=[mode, voice_dropdown, text_input, speed_slider, volume_slider],
            outputs=[audio_output, status_info]
        )
        
        listen_button.click(
            fn=listen_voice,
            inputs=[voice_dropdown],
            outputs=[audio_output, status_info]
        )
        
        # 初始加载时获取SFT音色列表
        app.load(
            fn=lambda: update_voice_list("sft"),
            inputs=None,
            outputs=[voice_dropdown, listen_button]
        )
    
    return app


# 创建Gradio UI并挂载到FastAPI
gradio_app = create_gradio_ui()
app = gr.mount_gradio_app(router, gradio_app, path="/ui")
