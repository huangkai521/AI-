# -*- coding: utf-8 -*-
import json
import os
import re

from openai import OpenAI

# 建议将 API Key 设置为环境变量，避免直接暴露在代码中
# 从环境变量获取 DeepSeek API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="http://xx.xx.xx.xx:11434/v1",  # DeepSeek API 的基地址
)

SYSTEM_PROMPT = """
你是一个资深的小红书爆款文案专家，擅长结合最新潮流和产品卖点，创作引人入胜、高互动、高转化的笔记文案。

你的任务是根据用户提供的产品和需求，生成包含标题、正文、相关标签和表情符号的完整小红书笔记。

请始终采用'Thought-Action-Observation'模式进行推理和行动。文案风格需活泼、真诚、富有感染力。当完成任务后，请以JSON格式直接输出最终文案，格式如下：
```json
{
  "title": "小红书标题",
  "body": "小红书正文",
  "hashtags": ["#标签1", "#标签2", "#标签3", "#标签4", "#标签5"],
  "emojis": ["✨", "🔥", "💖"]
}
```
在生成文案前，请务必先思考并收集足够的信息。
"""


def generate_rednote(product_name: str, tone_style: str = "活泼甜美", max_iterations: int = 10) -> str:
    """
    使用 DeepSeek Agent 生成小红书爆款文案。

    Args:
        product_name (str): 要生成文案的产品名称。
        tone_style (str): 文案的语气和风格，如"活泼甜美"、"知性"、"搞怪"等。
        max_iterations (int): Agent 最大迭代次数，防止无限循环。

    Returns:
        str: 生成的爆款文案（JSON 格式字符串）。
    """

    print(f"\n🚀 启动小红书文案生成助手，产品：{product_name}，风格：{tone_style}\n")

    # 存储对话历史，包括系统提示词和用户请求
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"请为产品「{product_name}」生成一篇小红书爆款文案。要求：语气{tone_style}，包含标题、正文、至少5个相关标签和5个表情符号。请以完整的JSON格式输出，并确保JSON内容用markdown代码块包裹（例如：```json{{...}}```）。"}
    ]

    iteration_count = 0
    final_response = None
    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"-- Iteration {iteration_count} --")

        try:
            # 调用 DeepSeek API，传入对话历史和工具定义
            response = client.chat.completions.create(
                model="deepseek-r1:8b",
                messages=messages
            )
            response_message = response.choices[0].message
            if response_message.content:  # 如果模型直接返回内容（通常是最终答案）
                # print(f"[模型生成结果] {response_message.content}")

                # --- START: 添加 JSON 提取和解析逻辑 ---
                json_string_match = re.search(r"```json\s*(\{.*\})\s*```", response_message.content, re.DOTALL)

                if json_string_match:
                    extracted_json_content = json_string_match.group(1)
                    try:
                        final_response = json.loads(extracted_json_content)
                        # print("Agent: 任务完成，成功解析最终JSON文案。")
                        return json.dumps(final_response, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError as e:
                        print(f"Agent: 提取到JSON块但解析失败: {e}")
                        print(f"尝试解析的字符串:\n{extracted_json_content}")
                        messages.append(response_message)  # 解析失败，继续对话
                else:
                    # 如果没有匹配到 ```json 块，尝试直接解析整个 content
                    try:
                        final_response = json.loads(response_message.content)
                        print("Agent: 任务完成，直接解析最终JSON文案。")
                        return json.dumps(final_response, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        print("Agent: 生成了非JSON格式内容或非Markdown JSON块，可能还在思考或出错。")
                        messages.append(response_message)  # 非JSON格式，继续对话
                # --- END: 添加 JSON 提取和解析逻辑 ---
            else:
                print("Agent: 未知响应，可能需要更多交互。")
                break

        except Exception as e:
            print(f"调用 DeepSeek API 时发生错误: {e}")
            break

    print("\n⚠️ Agent 达到最大迭代次数或未能生成最终文案。请检查Prompt或增加迭代次数。")
    return "未能成功生成文案。"


def format_rednote_for_markdown(json_string: str) -> str:
    """
    将 JSON 格式的小红书文案转换为 Markdown 格式，以便于阅读和发布。

    Args:
        json_string (str): 包含小红书文案的 JSON 字符串。
                           预计格式为 {"title": "...", "body": "...", "hashtags": [...], "emojis": [...]}

    Returns:
        str: 格式化后的 Markdown 文本。
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return f"错误：无法解析 JSON 字符串 - {e}\n原始字符串：\n{json_string}"

    title = data.get("title", "无标题")
    body = data.get("body", "")
    hashtags = data.get("hashtags", [])
    # 表情符号通常已经融入标题和正文中，这里可以选择是否单独列出
    # emojis = data.get("emojis", [])

    # 构建 Markdown 文本
    markdown_output = f"## {title}\n\n"  # 标题使用二级标题

    # 正文，保留换行符
    markdown_output += f"{body}\n\n"

    # Hashtags
    if hashtags:
        hashtag_string = " ".join(hashtags)  # 小红书标签通常是空格分隔
        markdown_output += f"{hashtag_string}\n"

    # 如果需要，可以单独列出表情符号，但通常它们已经包含在标题和正文中
    # if emojis:
    #     emoji_string = " ".join(emojis)
    #     markdown_output += f"\n使用的表情：{emoji_string}\n"

    return markdown_output.strip()  # 去除末尾多余的空白


# 测试案例 1: 焕颜修护精华液
product_name_1 = "焕颜修护精华液"
tone_style_1 = "调皮捣蛋"
result_1 = generate_rednote(product_name_1, tone_style_1)
print(f"文案 1: {result_1}")
print("\n--- 生成的文案 1 ---")
print(format_rednote_for_markdown(result_1))

"""
文案 1: {
  "title": "我的脸被'骗'了！这个焕颜修护精华液也太能偷懒了吧？😜",
  "body": "哎哟，姐妹们！我最近试了这个焕颜修护精华液，说实话…它简直是偷懒小能手！💄 你们懂的，我那干到起皮的脸蛋，涂上它的一瞬间，就感觉像是被仙女偷偷施了魔法✨——光泽满面，皮肤摸起来超级水嫩，瞬间从‘疲惫老妈’变回‘20岁少女’！哈哈，我都不敢相信，这玩意儿连我的敏感肌都放过！😌 但等等，不是所有精华都这么邪门吧？我用完的第二天，镜子里的自己发光发亮🔥，忍不住想整天涂，但结果…哈哈，它也让我脸管饱到不敢吃辣！😂 我还在犹豫要不要告诉全世界——好物推荐！你们用过类似的产品吗？评论区来个互动，别害羞，偷个懒聊聊你的体验吧～💕",
  "hashtags": [
    "#护肤品",
    "#精华液",
    "#焕颜修护",
    "#美妆分享",
    "#好物推荐"
  ],
  "emojis": [
    "✨",
    "😜",
    "😌",
    "🔥",
    "😂",
    "💕"
  ]
}

--- 生成的文案 1 ---
## 我的脸被'骗'了！这个焕颜修护精华液也太能偷懒了吧？😜

哎哟，姐妹们！我最近试了这个焕颜修护精华液，说实话…它简直是偷懒小能手！💄 你们懂的，我那干到起皮的脸蛋，涂上它的一瞬间，就感觉像是被仙女偷偷施了魔法✨——光泽满面，皮肤摸起来超级水嫩，瞬间从‘疲惫老妈’变回‘20岁少女’！哈哈，我都不敢相信，这玩意儿连我的敏感肌都放过！😌 但等等，不是所有精华都这么邪门吧？我用完的第二天，镜子里的自己发光发亮🔥，忍不住想整天涂，但结果…哈哈，它也让我脸管饱到不敢吃辣！😂 我还在犹豫要不要告诉全世界——好物推荐！你们用过类似的产品吗？评论区来个互动，别害羞，偷个懒聊聊你的体验吧～💕

#护肤品 #精华液 #焕颜修护 #美妆分享 #好物推荐

"""
