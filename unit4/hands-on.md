# 직접 해보기

이제 최종 에이전트 제작에 더 깊이 뛰어들 준비가 되었으니, 검토를 위해 에이전트를 제출하는 방법을 살펴보겠습니다.

## 데이터셋

이 리더보드에 사용된 데이터셋은 GAIA의 **검증** 세트 레벨 1 질문에서 추출된 20개의 질문으로 구성됩니다.

선택된 질문은 질문에 답하는 데 필요한 도구 및 단계 수를 기준으로 필터링되었습니다.

GAIA 벤치마크의 현재 모습을 바탕으로, 레벨 1 질문에서 30%를 목표로 하는 것이 공정한 테스트라고 생각합니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit4/leaderboard%20GAIA%2024%3A04%3A2025.png" alt="GAIA 현재 상태!" />

## 프로세스

이제 여러분의 마음속에는 아마도 "어떻게 제출을 시작하나요?"라는 큰 질문이 있을 것입니다.

이 유닛을 위해, 우리는 질문을 가져오고 채점을 위해 답변을 보낼 수 있는 API를 만들었습니다.
다음은 경로 요약입니다 (대화형 세부 정보는 [라이브 문서](https://agents-course-unit4-scoring.hf.space/docs) 참조):

*   **`GET /questions`**: 필터링된 전체 평가 질문 목록을 검색합니다.
*   **`GET /random-question`**: 목록에서 단일 무작위 질문을 가져옵니다.
*   **`GET /files/{task_id}`**: 주어진 작업 ID와 관련된 특정 파일을 다운로드합니다.
*   **`POST /submit`**: 에이전트 답변을 제출하고, 점수를 계산하며, 리더보드를 업데이트합니다.

제출 함수는 답변을 정답과 **정확히 일치**하는 방식으로 비교하므로, 프롬프트를 잘 작성하세요! GAIA 팀은 에이전트를 위한 프롬프트 예시를 [여기](https://huggingface.co/spaces/gaia-benchmark/leaderboard)에 공유했습니다 (이 과정에서는 제출물에 "FINAL ANSWER" 텍스트를 포함하지 않도록 하세요. 에이전트가 답변만 회신하도록 하세요).

🎨 **템플릿을 나만의 것으로 만드세요!**

API와 상호 작용하는 과정을 보여주기 위해, 우리는 시작점으로 [기본 템플릿](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)을 포함했습니다.

자유롭게—그리고 **적극 권장**—변경하거나, 추가하거나, 완전히 재구성하세요! 여러분의 접근 방식과 창의성에 가장 잘 맞도록 어떤 방식으로든 수정하세요.

이 템플릿을 제출하려면 API에 필요한 3가지 사항을 계산해야 합니다:

*   **사용자 이름:** 제출을 식별하는 데 사용되는 Hugging Face 사용자 이름 (여기서는 Gradio 로그인으로 얻음).
*   **코드 링크 (`agent_code`):** 확인 목적으로 Hugging Face Space 코드 (`.../tree/main`)로 연결되는 URL이므로, 스페이스를 공개 상태로 유지하세요.
*   **답변 (`answers`):** 채점을 위해 에이전트가 생성한 응답 목록 (`{"task_id": ..., "submitted_answer": ...}`).

따라서 이 [템플릿](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)을 자신의 Hugging Face 프로필에 복제하여 시작하는 것을 권장합니다.

🏆 리더보드는 [여기](https://huggingface.co/spaces/agents-course/Students_leaderboard)에서 확인하세요.

*친근한 참고: 이 리더보드는 재미를 위한 것입니다! 전체 검증 없이 점수를 제출하는 것이 가능하다는 것을 알고 있습니다. 공개 링크 없이 너무 많은 높은 점수가 게시되는 것을 보면, 리더보드를 유용하게 유지하기 위해 일부 항목을 검토, 조정 또는 제거해야 할 수도 있습니다.*
리더보드에는 여러분의 스페이스 코드베이스 링크가 표시됩니다. 이 리더보드는 학생들만을 위한 것이므로, 자랑스러운 점수를 얻었다면 스페이스를 공개 상태로 유지하세요.
<iframe
	src="https://agents-course-students-leaderboard.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>