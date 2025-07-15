import os
import sys
import yaml
import threading
import time
import traceback
import glob
import json
import ast
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, List, Optional, Generator, Tuple, Any
import gradio as gr

# VideoLLaMA3 核心导入
script_dir = Path(__file__).parent
root_dir = script_dir / 'VideoLLaMA3'
sys.path.insert(0, str(root_dir))

# 定义Gradio界面顶部的HTML内容
HEADER = ("""
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <div>
    <h1>FireVED：火灾视频应急事件检测</h1>
    <h5 style="margin: 0;">🔥Gradio Demo for Fire Video Emergency Detection🔥</h5>
  </div>
</div>
""")
question = "<video>\n请你分析视频片段，判断其中是否发生了与火灾或其他突发情况相关的应急事件。请指出各事件其在视频片段中的时间范围，并简要描述事件内容。若存在多个不同的事件，请按时间顺序输出多个json单元；当视频中发生新的、有意义的、与火灾相关的事件，或当前事件状态发生显著变化时，请开始一个新的事件段。事件段可以重叠。\n每个事件输出要求如下（严格遵循 JSON 格式）：\n{'emergency_exist': '是' 或 '否',  // 是否发生应急事件\n  'event_time': '起始秒-结束秒',  // 时间范围，单位为秒，保留一位小数\n  'event_des': '事件简要描述'  }\n注意事项：时间是相对于该视频片段的局部时间（即片段起点为 0s）；所有事件的范围交集要求覆盖全时段。请仅输出符合上述格式的 JSON。当出现正常、异常事件切换时才区分事件输出，连续相同事件合并为一个事件表述，事件表述尽可能短。有烟雾也是异常。"

try:
    from VideoLLaMA3.videollama3 import disable_torch_init, model_init, mm_infer
    from VideoLLaMA3.videollama3.mm_utils import load_video, load_images, torch
except ImportError as e:
    print(f"警告: VideoLLaMA3 模块导入失败: {e}")
    print("请确保 VideoLLaMA3 目录存在且包含必要文件")

# =================== 配置段 ===================
class Config:
    """轻量化配置管理"""
    DEFAULT_CONFIG = {
        "model": {
            "path": "model_ck",
            "device": "cuda",
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2"
        },
        "inference": {
            "fps": 1, # 尽量不要配置
            "max_frames": 160,
            "modal": "video",
            "max_new_tokens": 180,
            "do_sample": False,
            "merge_size": 2,
            "timeout": 300  # 5分钟超时
        }
    }

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """加载配置，失败时使用默认值"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    return user_config
            else:
                # 创建默认配置文件
                self.save_config(self.DEFAULT_CONFIG)
                return self.DEFAULT_CONFIG
        except Exception as e:
            print(f"配置加载失败，使用默认配置: {e}")
            return self.DEFAULT_CONFIG

    def save_config(self, config: Dict):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"配置保存失败: {e}")

    def reload_config(self) -> str:
        """重新加载配置"""
        try:
            self.config = self.load_config()
            return "配置已重新加载"
        except Exception as e:
            return f"配置重载失败: {e}"

# =================== 核心业务段 ===================
class VideoLLaMA3App:
    """核心应用类 - 集成所有功能"""

    def __init__(self):
        self.config = Config()
        self.model = None
        self.processor = None
        self.model_status = "未加载"
        self.conversation_history = []
        self.current_task = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def load_model(self) -> Tuple[bool, str]:
        """基于infer.py的模型加载逻辑"""
        try:
            print("开始加载模型...")

            # 严格按照infer.py的初始化方式
            disable_torch_init()

            # 使用infer.py完全相同的加载方式
            self.model, self.processor = model_init(
                model_path=self.config.config['model']['path'],
                torch_dtype=torch.bfloat16,
                attn_implementation=self.config.config['model']['attn_implementation']
            )

            device = self.config.config['model']['device']
            self.model = self.model.to(device)

            self.model_status = "已加载"
            print("模型加载成功")
            return True, "模型加载成功"

        except Exception as e:
            self.model_status = "加载失败"
            error_msg = f"模型加载失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return False, error_msg

    def unload_model(self) -> str:
        """内存清理"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model_status = "未加载"
            return "模型已卸载"
        except Exception as e:
            return f"模型卸载失败: {str(e)}"

    def stream_inference(self, video_path: str, question: str) -> Generator[str, None, None]:
        """流式
        - 基于mm_infer"""
        if self.model is None or self.processor is None:
            yield "❌ 模型未加载，请先加载模型"
            return

        try:
            # 状态更新
            yield "🎬 正在加载视频..."

            # 严格按照infer.py的处理方式
            frames, timestamps = load_video(
                video_path,
                fps=self.config.config['inference']['fps'],
                max_frames=self.config.config['inference']['max_frames']
            )

            yield f"📊 视频加载完成，共 {len(frames)} 帧"

            # 构建conversation，严格按照infer.py的格式
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                        {"type": "text", "text": question},
                    ]
                }
            ]

            yield "🧠 正在预处理数据..."

            # 使用完全相同的处理器和分析方式
            modal = self.config.config['inference']['modal']

            # 手动应用聊天模板以消除UserWarning
            # 处理器内部期望一个格式化的字符串，而不是对话列表
            text_input = self.processor.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                images=[frames] if modal != "text" else None,
                text=text_input, # 传入预格式化的字符串
                merge_size=self.config.config['inference']['merge_size'],
                return_tensors="pt",
            )

            yield "⚡ 正在分析中..."

            # 添加超时控制
            def run_inference():
                return mm_infer(
                    inputs,
                    model=self.model,
                    tokenizer=self.processor.tokenizer,
                    do_sample=self.config.config['inference']['do_sample'],
                    modal=modal,
                    max_new_tokens=self.config.config['inference']['max_new_tokens']
                )

            # 使用线程池执行分析
            self.current_task = self.executor.submit(run_inference)

            try:
                timeout = self.config.config['inference']['timeout']
                result = self.current_task.result(timeout=timeout)
                yield f"✅ 分析完成\n\n{result}"

                # 添加到会话历史（使用简化的用户消息）
                self.add_message("user", "开始视频分析")
                self.add_message("assistant", result)

            except TimeoutError:
                if self.current_task:
                    self.current_task.cancel()
                yield "⏰ 分析超时，已中断"

        except Exception as e:
            error_msg = f"❌ 分析失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            yield error_msg
        finally:
            self.current_task = None

    def interrupt_inference(self) -> str:
        """中断分析"""
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            return "分析已中断"
        return "没有正在进行的分析"

    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def clear_conversation(self):
        """清空对话历史"""
        self.conversation_history = []
        return "对话历史已清空"

    def get_model_info(self) -> str:
        """获取模型信息"""
        if self.model is None:
            return "模型未加载"

        try:
            model_path = self.config.config['model']['path']
            device = self.config.config['model']['device']
            return f"模型路径: {model_path}\n设备: {device}\n状态: {self.model_status}"
        except Exception as e:
            return f"获取模型信息失败: {str(e)}"

def parse_model_output(output_str: str) -> List[Dict]:
    """解析模型输出的字符串，提取事件字典"""
    events = []
    # 去除可能的Markdown代码块标记
    output_str = re.sub(r"```json\n?|```", "", output_str).strip()

    # 兼容单字典和多字典场景
    # 查找所有被大括号包围的部分
    # 这个正则表达式假设事件描述中不包含未转义的大括号
    dict_strings = re.findall(r"\{[^{}]+\}", output_str.replace("\n", ""))

    if not dict_strings:
        # 如果找不到，尝试将整个字符串作为单个Python字面量解析
        try:
            # 将Python字典转为JSON字符串，然后再解析
            # 这样可以处理单引号等问题
            temp_list = ast.literal_eval(f"[{output_str}]")
            events.extend(temp_list)
        except (ValueError, SyntaxError) as e:
            print(f"解析整个字符串失败: {e}")
            return [] # 如果还是失败，返回空列表
        return events

    for d_str in dict_strings:
        try:
            # ast.literal_eval 更安全，能处理Python的字典格式（如单引号）
            event = ast.literal_eval(d_str)
            if isinstance(event, dict):
                events.append(event)
        except (ValueError, SyntaxError) as e:
            print(f"解析单个事件失败: '{d_str}', 错误: {e}")
            continue
    return events


def format_timeline_output(result_str: str) -> str:
    """将模型输出的事件字符串格式化为包含状态和时间轴的HTML"""
    events = parse_model_output(result_str)

    if not events:
        status_md = "## <p style='text-align:center;'>⚪ 分析完成，未检测到有效事件信息。</p>"
        timeline_md = "<p>未检测到有效事件信息。</p>"
        return status_md + timeline_md

    # a. 总体状态
    is_emergency = any(item.get('emergency_exist') == '是' for item in events)
    if is_emergency:
        status_md = "## <p style='color:#FF4136; text-align:center;'>🔴 检测到应急事件！</p>"
    else:
        status_md = "## <p style='color:#2ECC40; text-align:center;'>🟢 未发现异常</p>"

    # b. 事件时间轴
    timeline_md = ""
    for i, event in enumerate(events, 1):
        if event.get('emergency_exist') == '是':
            style = "border-left: 5px solid #FF4136; padding-left: 15px; margin: 20px 0;"
            icon = "🔥"
            level = "警告"
        else:
            style = "border-left: 5px solid #2ECC40; padding-left: 15px; margin: 20px 0;"
            icon = "✅"
            level = "正常"

        timeline_md += f"""
        <div style='{style}'>
            <p style='font-size: 1.1em; font-weight: bold; margin:0;'>{icon} 事件 {i}: {level}</p>
            <p style='margin: 5px 0;'><strong>时间范围:</strong> {event.get('event_time', 'N/A')}</p>
            <p style='margin: 5px 0;'><strong>事件描述:</strong> {event.get('event_des', 'N/A')}</p>
        </div>
        """
    return status_md + timeline_md


# =================== UI段 ===================
def create_gradio_interface(app: VideoLLaMA3App) -> gr.Blocks:
    """创建简洁高效的UI界面"""

    gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

    with gr.Blocks(
        title="FireVED: Fire Video Emergency Detection",
        theme=gr.themes.Soft()
    ) as demo:
        gr.HTML(HEADER)

        # 状态变量
        conversation_state = gr.State([])

        with gr.Row():
            # 左侧控制区域 (scale=2)
            with gr.Column(scale=2):
                with gr.Tab(label="Input"):
                    # 输入区域
                    video_input = gr.Video(
                        label="上传视频",
                        container=True,
                        interactive=True,
                        height=400,
                        sources=["upload"],
                    )

                    # 按钮组
                    with gr.Row():
                        submit_btn = gr.Button("检测", variant="primary")
                        interrupt_btn = gr.Button("中断", variant="stop")

                    # 状态显示
                    status_display = gr.Textbox(
                        label="系统状态",
                        value="就绪",
                        interactive=False,
                        max_lines=2
                    )

                    # 对话控制
                    with gr.Row():
                        clear_btn = gr.Button("清空对话", size="sm")

                    # 视频案例
                    gr.Markdown("### Examples")

                    # 检索 examples 目录下所有 mp4 文件，生成示例文件列表
                    examples_dir = "examples"
                    example_videos = glob.glob(os.path.join(examples_dir, "*.mp4"))
                    examples_data = []
                    for video_path in example_videos:
                        examples_data.append([
                            video_path
                        ])

                    if examples_data:
                        examples = gr.Examples(
                            examples=examples_data,
                            inputs=[video_input],
                            label="点击选择示例"
                        )
                    else:
                        gr.Markdown("*没有找到示例视频文件*")

                with gr.Tab(label="Configure"):
                    # 模型管理
                    gr.Markdown("### 模型管理")
                    model_status = gr.Textbox(
                        label="模型状态",
                        value="未加载",
                        interactive=False
                    )

                    with gr.Row():
                        load_btn = gr.Button("加载模型", size="sm")
                        unload_btn = gr.Button("卸载模型", size="sm")

                    # 配置管理
                    gr.Markdown("### 配置管理")
                    config_display = gr.Code(
                        value=yaml.dump(app.config.config, default_flow_style=False),
                        language="yaml",
                        label="当前配置",
                        lines=8
                    )

                    reload_config_btn = gr.Button("重载配置", size="sm")

            # 右侧聊天区域
            with gr.Column(scale=2):
                with gr.Blocks():
                    chatbot = gr.Chatbot(
                        type="messages",
                        height=514,
                        bubble_full_width=False,
                        show_copy_button=True,
                        label="对话历史"
                    )
                with gr.Blocks():
                    timeline_output = gr.Markdown("### <p style='text-align:center;'>⚪ 等待分析...</p>", container=True)

        # 事件处理函数
        def handle_submit(video, history):
            """处理提交事件"""
            if not video:
                gr.Warning("请上传视频文件")
                return history, history, "### <p style='text-align:center;'>⚪ 等待分析...</p>", "请上传视频文件"

            if app.model_status != "已加载":
                gr.Warning("模型未加载，请先加载模型")
                return history, history, "### <p style='text-align:center;'>⚪ 等待分析...</p>", "请先加载模型"

            # 清空上一次的结果
            initial_timeline_md = "## <p style='text-align:center;'>⏳ 分析进行中...</p>"

            # 添加用户消息（显示简化信息）
            user_msg = {"role": "user", "content": "视频分析任务已提交，请稍候..."}
            history.append(user_msg)

            # 添加助手消息占位符
            assistant_msg = {"role": "assistant", "content": "⏳ 开始分析，请稍候..."}
            history.append(assistant_msg)
            yield history, history, initial_timeline_md, "分析中..."

            final_result = ""
            # 流式分析（使用内部问题）
            for partial_response in app.stream_inference(video, question):
                history[-1]["content"] = partial_response
                final_result = partial_response
                yield history, history, initial_timeline_md, "分析中..."

            # 分析完成后，格式化最终结果
            prefix = "✅ 分析完成\n\n"
            if final_result.startswith(prefix):
                json_part = final_result[len(prefix):].strip()

                # 1. 为 timeline_output 生成HTML
                combined_md = format_timeline_output(json_part)

                # 2. 为Chatbot准备格式化的JSON
                try:
                    events = parse_model_output(json_part)
                    # 确保即使只有一个事件，也以列表形式出现
                    if isinstance(events, dict):
                       events = [events]
                    pretty_json = json.dumps(events, indent=2, ensure_ascii=False)
                    chatbot_content = f"```json\n{pretty_json}\n```"
                except Exception:
                    chatbot_content = f"```json\n{json_part}\n```"

                history[-1]["content"] = chatbot_content
                yield history, history, combined_md, "分析完成"

            else:
                # 对于错误或超时等情况，直接显示
                history[-1]["content"] = final_result
                error_md = f"## <p style='color:red;text-align:center;'>❌ 分析失败或超时</p><p>错误信息: {final_result}</p>"
                yield history, history, error_md, "分析失败"

        def handle_load_model():
            """处理模型加载"""
            success, message = app.load_model()
            status = "已加载" if success else "加载失败"
            info = app.get_model_info()
            return status, info, message

        def handle_unload_model():
            """处理模型卸载"""
            message = app.unload_model()
            info = app.get_model_info()
            return "未加载", info, message

        def handle_interrupt():
            """处理中断事件"""
            message = app.interrupt_inference()
            return message

        def handle_clear_conversation(history):
            """处理清空对话"""
            app.clear_conversation()
            return [], [], "### <p style='text-align:center;'>⚪ 等待分析...</p>", "对话历史已清空"

        def handle_reload_config():
            """处理配置重载"""
            message = app.config.reload_config()
            new_config = yaml.dump(app.config.config, default_flow_style=False)
            return new_config, message

        # 绑定事件
        submit_btn.click(
            fn=handle_submit,
            inputs=[video_input, conversation_state],
            outputs=[chatbot, conversation_state, timeline_output, status_display]
        )

        load_btn.click(
            fn=handle_load_model,
            outputs=[model_status, config_display, status_display]
        )

        unload_btn.click(
            fn=handle_unload_model,
            outputs=[model_status, config_display, status_display]
        )

        interrupt_btn.click(
            fn=handle_interrupt,
            outputs=[status_display]
        )

        clear_btn.click(
            fn=handle_clear_conversation,
            inputs=[conversation_state],
            outputs=[chatbot, conversation_state, timeline_output, status_display]
        )

        reload_config_btn.click(
            fn=handle_reload_config,
            outputs=[config_display, status_display]
        )

    return demo

# =================== 主程序段 ===================
try:
    print("🚀 启动 VideoLLaMA3 Gradio 应用...")

    # 创建应用实例
    app = VideoLLaMA3App()

    # 创建界面
    demo = create_gradio_interface(app)

except Exception as e:
    print(f"❌ 应用启动失败: {e}")
    print(traceback.format_exc())

if __name__ == "__main__":
    # 启动应用
    print("🌐 启动Web界面...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False,
        debug=False,
        quiet=False,
        allowed_paths=["assets"]
    )