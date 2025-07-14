# 导入os库，用于处理文件和目录路径
import os
# 从os库中导入path模块，并重命名为osp，方便后续使用
import os.path as osp

# 导入gradio库，用于创建Web UI界面
import gradio as gr
# 导入spaces，用于在Hugging Face Spaces上进行部署优化，如此处的GPU时长控制
import spaces
# 导入torch库，PyTorch的核心库
import torch
# 从threading库导入Thread，用于在后台运行模型生成，避免UI阻塞
from threading import Thread
# 从transformers库导入所需的类：
# AutoModelForCausalLM: 用于加载预训练的因果语言模型
# AutoProcessor: 用于加载与模型匹配的预处理器，处理文本、图像、视频等多种输入
# TextIteratorStreamer: 用于实现流式文本输出
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer


# 定义Gradio界面顶部的HTML内容，包含项目标题、Logo、Github链接、论文链接等
HEADER = ("""
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <a href="https://github.com/DAMO-NLP-SG/VideoLLaMA3" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
    <img src="https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/logo.png?raw=true" alt="VideoLLaMA 3 🔥🚀🔥" style="max-width: 120px; height: auto;">
  </a>
  <div>
    <h1>VideoLLaMA 3: Frontier Multimodal Foundation Models for Video Understanding</h1>
    <h5 style="margin: 0;">If this demo please you, please give us a star ⭐ on Github or 💖 on this space.</h5>
  </div>
</div>

<div style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://github.com/DAMO-NLP-SG/VideoLLaMA3"><img src='https://img.shields.io/badge/Github-VideoLLaMA3-9C276A' style="margin-right: 5px;"></a>
  <a href="https://arxiv.org/pdf/2501.13106"><img src="https://img.shields.io/badge/Arxiv-2501.13106-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/collections/DAMO-NLP-SG/videollama3-678cdda9281a0e32fe79af15"><img src="https://img.shields.io/badge/🤗-Checkpoints-ED5A22.svg" style="margin-right: 5px;"></a>
  <a href="https://github.com/DAMO-NLP-SG/VideoLLaMA3/stargazers"><img src="https://img.shields.io/github/stars/DAMO-NLP-SG/VideoLLaMA3.svg?style=social"></a>
</div>
""")

# 设置模型运行的设备，优先使用CUDA（NVIDIA GPU）
device = "cuda"
# 加载预训练的VideoLLaMA3模型
model = AutoModelForCausalLM.from_pretrained(
    "DAMO-NLP-SG/VideoLLaMA3-2B",  # 模型在Hugging Face Hub上的名称
    trust_remote_code=True,  # 允许执行模型仓库中的自定义代码
    torch_dtype=torch.bfloat16,  # 使用bfloat16数据类型，以节省显存并加速计算
    attn_implementation="flash_attention_2",  # 使用Flash Attention 2优化，进一步提升速度和效率
)
# 将模型移动到指定的设备上
model.to(device)
# 加载与模型配套的预处理器
processor = AutoProcessor.from_pretrained("DAMO-NLP-SG/VideoLLaMA3-7B", trust_remote_code=True)


# 定义存放示例文件的目录
example_dir = "./examples"
# 定义支持的图片和视频文件格式
image_formats = ("png", "jpg", "jpeg")
video_formats = ("mp4",)

# 初始化用于存储示例文件路径的列表
image_examples, video_examples = [], []
# 如果示例目录存在，则加载其中的文件
if example_dir is not None:
    # 获取目录下的所有文件名
    example_files = [
        osp.join(example_dir, f) for f in os.listdir(example_dir)
    ]
    # 遍历文件，根据后缀名将其分类到图片或视频示例列表中
    for example_file in example_files:
        if example_file.endswith(image_formats):
            image_examples.append([example_file])
        elif example_file.endswith(video_formats):
            video_examples.append([example_file])


# 定义视频上传后的回调函数
def _on_video_upload(messages, video):
        """
        当用户上传视频时，此函数被调用。
        它会将视频路径添加到一个代表对话历史的列表中。

        参数:
            messages (list): 当前的对话历史列表。
            video (str): 上传视频的本地文件路径。

        返回:
            tuple: 更新后的对话历史列表和一个None值（用于清空视频上传组件）。
        """
        if video is not None:
            # messages.append({"role": "user", "content": gr.Video(video)})
            # 将视频路径封装成一个字典，添加到对话历史中
            messages.append({"role": "user", "content": {"path": video}})
        return messages, None

# 定义图片上传后的回调函数
def _on_image_upload(messages, image):
    """
    当用户上传图片时，此函数被调用。
    它会将图片路径添加到一个代表对话历史的列表中。

    参数:
        messages (list): 当前的对话历史列表。
        image (str): 上传图片的本地文件路径。

    返回:
        tuple: 更新后的对话历史列表和一个None值（用于清空图片上传组件）。
    """
    if image is not None:
        # messages.append({"role": "user", "content": gr.Image(image)})
        # 将图片路径封装成一个字典，添加到对话历史中
        messages.append({"role": "user", "content": {"path": image}})
    return messages, None

# 定义文本提交后的回调函数
def _on_text_submit(messages, text):
    """
    当用户提交文本时，此函数被调用。
    它会将文本消息添加到一个代表对话历史的列表中。

    参数:
        messages (list): 当前的对话历史列表。
        text (str): 用户输入的文本。

    返回:
        tuple: 更新后的对话历史列表和一个空字符串（用于清空文本输入框）。
    """
    messages.append({"role": "user", "content": text})
    return messages, ""

# 使用Hugging Face Spaces的装饰器，为该函数申请GPU资源，运行时长限制为120秒
@spaces.GPU(duration=120)
def _predict(messages, input_text, do_sample, temperature, top_p, max_new_tokens,
             fps, max_frames):
    """
    核心预测函数，用于生成模型的回复。

    参数:
        messages (list): 完整的对话历史。
        input_text (str): 用户在文本框中最新输入的文本。
        do_sample (bool): 是否使用采样生成。
        temperature (float): 控制生成文本随机性的温度参数。
        top_p (float): Top-p (nucleus) 采样的阈值。
        max_new_tokens (int): 生成新token的最大数量。
        fps (int): 处理视频时抽取的帧率。
        max_frames (int): 处理视频时抽取的最大帧数。

    返回:
        generator: 一个生成器，逐个token地产生回复，实现流式输出。
    """
    # 如果文本框中有新的输入，先将其加入对话历史
    if len(input_text) > 0:
        messages.append({"role": "user", "content": input_text})
    # 初始化一个新的消息列表，用于传递给处理器
    new_messages = []
    # 初始化一个内容列表，用于合并连续的用户文本消息
    contents = []
    # 遍历原始对话历史，将其转换为模型处理器所需的格式
    for message in messages:
        if message["role"] == "assistant":
            # 如果遇到助手消息，先将之前收集的用户内容添加到新消息列表
            if len(contents):
                new_messages.append({"role": "user", "content": contents})
                contents = []
            # 然后添加助手消息
            new_messages.append(message)
        elif message["role"] == "user":
            # 如果是用户消息，判断内容是文本还是文件
            if isinstance(message["content"], str):
                # 如果是文本，添加到当前内容列表
                contents.append(message["content"])
            else:
                # 如果是文件（视频或图片），提取路径
                media_path = message["content"]["path"]
                if media_path.endswith(video_formats):
                    # 如果是视频，构建视频描述字典
                    contents.append({"type": "video", "video": {"video_path": media_path, "fps": fps, "max_frames": max_frames}})
                elif media_path.endswith(image_formats):
                    # 如果是图片，构建图片描述字典
                    contents.append({"type": "image", "image": {"image_path": media_path}})
                else:
                    raise ValueError(f"Unsupported media type: {media_path}")

    # 如果遍历结束后仍有未处理的用户内容，添加到新消息列表
    if len(contents):
        new_messages.append({"role": "user", "content": contents})

    # 如果没有用户输入，或者最后一条消息不是用户的，则不进行生成
    if len(new_messages) == 0 or new_messages[-1]["role"] != "user":
        return messages

    # 配置生成参数
    generation_config = {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens
    }

    # 使用处理器处理对话历史，生成模型输入
    inputs = processor(
        conversation=new_messages,
        add_system_prompt=True,      # 添加系统提示
        add_generation_prompt=True,  # 添加生成提示
        return_tensors="pt"          # 返回PyTorch张量
    )
    # 将输入数据移动到指定的设备（GPU）
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    # 如果输入中包含像素值（来自图片或视频），将其转换为bfloat16类型
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # 初始化流式输出器
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 准备传递给模型generate方法的参数
    generation_kwargs = {
        **inputs,
        **generation_config,
        "streamer": streamer,
    }

    # 创建并启动一个新线程来运行模型生成，避免阻塞UI
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 在对话历史中添加一个空的助手回复，用于后续填充
    messages.append({"role": "assistant", "content": ""})
    # 迭代流式输出器，获取生成的每个token
    for token in streamer:
        # 将新生成的token追加到最后一条助手消息的内容中
        messages[-1]['content'] += token
        # 使用yield返回更新后的对话历史，Gradio会自动更新UI
        yield messages


# 使用gr.Blocks()创建一个Gradio界面
with gr.Blocks() as interface:
    # 显示HTML头部
    gr.HTML(HEADER)
    # 创建一个水平布局
    with gr.Row():
        # 创建聊天机器人窗口
        chatbot = gr.Chatbot(type="messages", elem_id="chatbot", height=835)

        # 创建一个垂直布局，用于放置输入和配置组件
        with gr.Column():
            # 创建一个选项卡布局
            with gr.Tab(label="Input"):

                # 创建一个水平布局用于放置视频和图片上传组件
                with gr.Row():
                    input_video = gr.Video(sources=["upload"], label="Upload Video")
                    input_image = gr.Image(sources=["upload"], type="filepath", label="Upload Image")

                # 创建文本输入框
                input_text = gr.Textbox(label="Input Text", placeholder="Type your message here and press enter to submit")

                # 创建提交按钮
                submit_button = gr.Button("Generate")

                # 创建示例区域，点击后会自动填充输入
                gr.Examples(examples=[
                    [f"examples/装运过程火灾.mp4"],
                    [f"examples/工厂爆燃.mp4"],
                    [f"examples/室外镁合金粉未爆炸.mp4"],
                ], inputs=[input_video, input_text], label="Video examples")

            # 创建配置选项卡
            with gr.Tab(label="Configure"):
                # 创建可折叠的生成配置区域
                with gr.Accordion("Generation Config", open=True):
                    do_sample = gr.Checkbox(value=True, label="Do Sample")
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, label="Top P")
                    max_new_tokens = gr.Slider(minimum=0, maximum=4096, value=2048, step=1, label="Max New Tokens")

                # 创建可折叠的视频配置区域
                with gr.Accordion("Video Config", open=True):
                    fps = gr.Slider(minimum=0.0, maximum=10.0, value=1, label="FPS")
                    max_frames = gr.Slider(minimum=0, maximum=256, value=180, step=1, label="Max Frames")

    # ---- 事件绑定 ----
    # 将视频上传组件的change事件绑定到_on_video_upload函数
    input_video.change(_on_video_upload, [chatbot, input_video], [chatbot, input_video])
    # 将图片上传组件的change事件绑定到_on_image_upload函数
    input_image.change(_on_image_upload, [chatbot, input_image], [chatbot, input_image])
    # 将文本输入框的submit事件（按回车）绑定到_on_text_submit函数
    input_text.submit(_on_text_submit, [chatbot, input_text], [chatbot, input_text])
    # 将提交按钮的click事件绑定到_predict函数
    submit_button.click(
        _predict,
        [
            chatbot, input_text, do_sample, temperature, top_p, max_new_tokens,
            fps, max_frames
        ],
        [chatbot],
    )


# 如果该脚本作为主程序运行，则启动Gradio界面
if __name__ == "__main__":
    interface.launch()