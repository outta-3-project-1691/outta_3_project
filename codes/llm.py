import pathlib
import textwrap
import google.generativeai as genai
import google.generativeai as genai

# [1] 빈칸을 작성하시오.
# API 키
GOOGLE_API_KEY = ''
genai.configure(api_key=GOOGLE_API_KEY)

# 모델 초기화
def LLM(text):
    model = genai.GenerativeModel('gemini-pro')

    while True:
        user_input = text

        if user_input=="q":
            break

        else:
            """
            [Example]
            User: I want to change the top of the woman to a blue sweatshirt and the bottom to a black skirt.
            You: {"top":["blue","sweatshirt"],"bottom":["black","skirt"]}
            """
            # [2] 빈칸을 작성하시오.
            # 예시와 같이 성능 향상을 위해 프롬프트 튜닝을 진행
            instruction = """
                Request Interpretation: When the user asks to modify the top and/or bottom clothing of a subject, interpret the user's input precisely, paying attention to the details of color and clothing type.
                Response Structure: Respond with a JSON-like string. The "top" key should contain an array with the color and type of the top clothing, and the "bottom" key should contain an array with the color and type of the bottom clothing.
                Accuracy: Always ensure that the colors and clothing types in your response exactly match the user's request.
                Partial Information Handling: If the user only specifies one part (either top or bottom), include only that part in the JSON response. Leave out the unspecified part entirely.
                Error Handling: If the user’s request is ambiguous or missing essential information, provide a clarification prompt instead of generating a potentially incorrect response.
                [Example]
                    User: I want to change the top of the woman to a blue sweatshirt and the bottom to a black skirt.
                    You: {"top":["blue","sweatshirt"],"bottom":["black","skirt"]}
            
            """
            prompt = user_input

            # 전체 프롬프트 생성 (instruction 포함)
            full_prompt = f"{instruction}\n{prompt}"
            response = model.generate_content(full_prompt).text.replace('\n', '').replace(' ', '')

            # 응답 출력
            return response
