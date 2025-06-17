
import pandas as pd
from openai import OpenAI

def prepare_prompt(user_question, df_summary, matched_names=None):
    """Tạo prompt gửi cho AI"""
    csv_data = df_summary.to_csv(index=False)
    extra_info = ""
    if matched_names:
        extra_info = f"(Đã dò gần đúng tên khách hàng: {matched_names})\n"
    prompt = f"""
{extra_info}
Bạn là trợ lý dữ liệu bảo hành, nhiệm vụ là đọc bảng dữ liệu nhỏ sau (dưới dạng csv) và trả lời câu hỏi bằng tiếng Việt ngắn gọn, dễ hiểu, có số liệu cụ thể.
Dữ liệu bảng:
{csv_data}
Câu hỏi: {user_question}
"""
    return prompt

def query_openai(user_question, df_summary, api_key, model="gpt-4o", matched_names=None):
    """Gửi câu hỏi + bảng dữ liệu nhỏ đến OpenAI, trả về câu trả lời"""
    if df_summary.empty:
        return "Không có dữ liệu phù hợp để trả lời.", None

    prompt = prepare_prompt(user_question, df_summary, matched_names)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip(), prompt
    except Exception as e:
        return f"Lỗi khi gọi OpenAI: {e}", None
