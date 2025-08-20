# GAIA란 무엇인가요?

[GAIA](https://huggingface.co/papers/2311.12983)는 추론, 다중 모드 이해, 웹 브라우징, 능숙한 도구 사용과 같은 핵심 역량의 조합을 요구하는 **실제 작업에서 AI 비서를 평가하도록 설계된 벤치마크**입니다.

이 벤치마크는 _"[GAIA: 일반 AI 비서를 위한 벤치마크](https://huggingface.co/papers/2311.12983)"_라는 논문에서 소개되었습니다.

이 벤치마크는 **인간에게는 개념적으로 간단하지만 현재 AI 시스템에는 매우 어려운 466개의 신중하게 선별된 질문**을 특징으로 합니다.

격차를 설명하자면:
- **인간**: 약 92% 성공률
- **플러그인 장착 GPT-4**: 약 15%
- **Deep Research (OpenAI)**: 검증 세트에서 67.36%

GAIA는 AI 모델의 현재 한계를 강조하고 진정한 범용 AI 비서로의 발전을 평가하기 위한 엄격한 벤치마크를 제공합니다.

## 🌱 GAIA의 핵심 원칙

GAIA는 다음 기둥을 중심으로 신중하게 설계되었습니다.

- 🔍 **실제 난이도**: 작업은 다단계 추론, 다중 모드 이해 및 도구 상호 작용을 요구합니다.
- 🧾 **인간 해석 가능성**: AI에게는 어렵지만, 작업은 개념적으로 간단하고 인간이 따라하기 쉽습니다.
- 🛡️ **비게임성**: 올바른 답변은 전체 작업 실행을 요구하므로 무차별 대입은 비효율적입니다.
- 🧰 **평가 용이성**: 답변은 간결하고 사실적이며 모호하지 않아 벤치마킹에 이상적입니다.

## 난이도 수준

GAIA 작업은 특정 기술을 테스트하는 **점점 더 복잡해지는 세 가지 수준**으로 구성됩니다.

- **레벨 1**: 5단계 미만 및 최소한의 도구 사용을 요구합니다.
- **레벨 2**: 여러 도구와 5-10단계 간의 더 복잡한 추론 및 조정을 포함합니다.
- **레벨 3**: 장기 계획 및 다양한 도구의 고급 통합을 요구합니다.

![GAIA levels](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit4/gaia_levels.png)

## 어려운 GAIA 질문의 예시

> 2008년 그림 "우즈베키스탄 자수"에 묘사된 과일 중 영화 "마지막 항해"의 떠다니는 소품으로 나중에 사용된 대양 정기선의 1949년 10월 아침 식사 메뉴에 포함된 것은 무엇입니까? 그림에서 12시 방향부터 시계 방향으로 배열된 순서대로 쉼표로 구분된 목록으로 항목을 제공하십시오. 각 과일의 복수형을 사용하십시오.

보시다시피, 이 질문은 여러 면에서 AI 시스템에 도전합니다.

- **구조화된 응답 형식**을 요구합니다.
- **다중 모드 추론**(예: 이미지 분석)을 포함합니다.
- 상호 의존적인 사실의 **다중 홉 검색**을 요구합니다.
  - 그림에서 과일 식별
  - *마지막 항해*에 사용된 대양 정기선 발견
  - 해당 선박의 1949년 10월 아침 식사 메뉴 조회
- 올바른 순서로 해결하기 위한 **정확한 순서 지정** 및 고수준 계획이 필요합니다.

이러한 종류의 작업은 독립형 LLM이 종종 부족한 부분을 강조하며, GAIA를 여러 단계와 모드에 걸쳐 추론, 검색 및 실행할 수 있는 **에이전트 기반 시스템**에 이상적인 벤치마크로 만듭니다.

![GAIA capabilities plot](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit4/gaia_capabilities.png)

## 실시간 평가

지속적인 벤치마킹을 장려하기 위해 **GAIA는 Hugging Face에서 호스팅되는 공개 리더보드를 제공**하며, 여기에서 **300개의 테스트 질문**에 대해 모델을 테스트할 수 있습니다.

👉 리더보드는 [여기](https://huggingface.co/spaces/gaia-benchmark/leaderboard)에서 확인하세요.

<iframe
	src="https://gaia-benchmark-leaderboard.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

GAIA에 대해 더 자세히 알아보고 싶으신가요?

- 📄 [전체 논문 읽기](https://huggingface.co/papers/2311.12983)
- 📄 [OpenAI의 Deep Research 출시 게시물](https://openai.com/index/introducing-deep-research/)
- 📄 [오픈 소스 DeepResearch – 검색 에이전트 해방](https://huggingface.co/blog/open-deep-research)