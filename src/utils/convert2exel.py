import pandas as pd

def display2exel(
    question_path: str = 'test/qna/question.txt',
    response_path: str = 'test/qna/response.txt',
    output_path: str = 'test/qna/output.xlsx'
):
    # Đọc file câu hỏi và câu trả lời
    with open(question_path, 'r', encoding='utf-8') as f:
        questions = f.readlines()

    with open(response_path, 'r', encoding='utf-8') as f:
        answers = f.readlines()

    # Tạo DataFrame
    data = {'Question': [q.strip() for q in questions], 'Response': [a.strip() for a in answers]}
    df = pd.DataFrame(data)

    # Lưu DataFrame thành file Excel
    df.to_excel(output_path, index=False)
