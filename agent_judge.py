"""검색 대화와 분리된 최종 답변 생성."""
from agent_config import FINAL_ANSWER_INSTRUCTIONS, LABEL, MODEL


def run_final_judge(client, question: str, collected_context: str) -> dict:
    """질문과 검색 근거만 전달하고 답변 형식을 확인한다.

    정답 라벨은 전달받지 않는다. 정답 비교는 run_agent.py에서 수행한다.
    """
    # tools와 previous_response_id 없이 독립적인 판정 호출을 만든다.
    response = client.responses.create(
        model=MODEL,
        instructions=FINAL_ANSWER_INSTRUCTIONS,
        input=(
            f"Question:\n{question}\n\n"
            f"Evidence:\n{collected_context}"
        ),
    )
    prediction = response.output_text.strip().lower()

    return {
        "response_id": response.id,
        "prediction": prediction,
        "raw_answer": response.output_text,
        "response_status": response.status,
        "is_valid": response.status == "completed" and prediction in LABEL,
    }
