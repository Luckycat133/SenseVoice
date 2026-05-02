import gradio as gr
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

print("Loading model from ModelScope...")
model_dir = "iic/SenseVoiceSmall"
try:
    m = SenseVoiceSmall(model_dir, "mac", device="cpu")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

def transcribe(audio_path):
    if not audio_path:
        return ""
    try:
        res = m.inference(data_in=audio_path)[0][0]["text"]
        return rich_transcription_postprocess(res)
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="上传音频 (Upload Audio)"),
    outputs=gr.Textbox(label="转录结果 (Transcription)", lines=10),
    title="SenseVoice 本地语音转文字",
    description="上传音频进行快速且高准度的语音转写（带情感和标签分析）。",
)

iface.launch(server_name="127.0.0.1", server_port=7860)
