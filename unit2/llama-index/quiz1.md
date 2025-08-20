# 작은 퀴즈 (미채점) [[quiz1]]

지금까지 LlamaIndex의 주요 구성 요소와 도구에 대해 논의했습니다.
**스스로 테스트하는 것**이 배우는 가장 좋은 방법이고 [능력 착각을 피하는](https://www.coursera.org/lecture/learning-how-to-learn/illusions-of-competence-BuFzf) 데 도움이 되므로 짧은 퀴즈를 풀어볼 시간입니다.
이는 **어떤 지식을 보강해야 하는지** 찾는 데 도움이 될 것입니다.

이 퀴즈는 선택 사항이며 채점되지 않습니다.

### Q1: QueryEngine이란 무엇인가요?
다음 중 QueryEngine 구성 요소를 가장 잘 설명하는 것은 무엇인가요?

<Question
choices={[
{
text: "검색 기능 없이 정적 텍스트만 처리하는 시스템.",
explain: "QueryEngine은 관련 정보를 검색하고 처리할 수 있어야 합니다.",
},
{
text: "RAG 프로세스의 일부로 관련 정보를 찾아 검색하는 구성 요소.",
explain: "이는 QueryEngine 구성 요소의 핵심 목적을 포착합니다.",
correct: true
},
{
text: "검색 기능 없이 벡터 임베딩만 저장하는 도구.",
explain: "QueryEngine은 임베딩을 저장하는 것 이상으로 정보를 적극적으로 검색하고 가져옵니다.",
},
{
text: "응답 품질만 평가하는 구성 요소.",
explain: "평가는 QueryEngine의 주요 검색 목적과 별개입니다.",
}
]}
/>

---

### Q2: FunctionTools의 목적은 무엇인가요?
FunctionTools가 Agent에게 중요한 이유는 무엇인가요?

<Question
choices={[
{
text: "대량의 데이터 저장을 처리하기 위해.",
explain: "FunctionTools는 주로 데이터 저장을 위한 것이 아닙니다.",
},
{
text: "Python 함수를 에이전트가 사용할 수 있는 도구로 변환하기 위해.",
explain: "FunctionTools는 Python 함수를 래핑하여 에이전트가 접근할 수 있도록 합니다.",
correct: true
},
{
text: "에이전트가 무작위 함수 정의를 생성하도록 허용하기 위해.",
explain: "FunctionTools는 함수를 에이전트가 사용할 수 있도록 하는 특정 목적을 수행합니다.",
},
{
text: "텍스트 데이터만 처리하기 위해.",
explain: "FunctionTools는 텍스트 처리뿐만 아니라 다양한 유형의 함수와 함께 작동할 수 있습니다.",
}
]}
/>

---

### Q3: LlamaIndex에서 Toolspecs란 무엇인가요?
Toolspecs의 주요 목적은 무엇인가요?

<Question
choices={[
{
text: "기능을 추가하지 않는 불필요한 구성 요소입니다.",
explain: "Toolspecs는 LlamaIndex 생태계에서 중요한 목적을 수행합니다.",
},
{
text: "에이전트 기능을 확장하는 커뮤니티 생성 도구 세트입니다.",
explain: "Toolspecs는 커뮤니티가 도구를 공유하고 재사용할 수 있도록 합니다.",
correct: true
},
{
text: "메모리 관리에만 사용됩니다.",
explain: "Toolspecs는 도구를 제공하는 것이지 메모리를 관리하는 것이 아닙니다.",
},
{
text: "텍스트 처리에서만 작동합니다.",
explain: "Toolspecs는 텍스트 처리뿐만 아니라 다양한 유형의 도구를 포함할 수 있습니다.",
}
]}
/>

---

### Q4: 도구를 생성하는 데 필요한 것은 무엇인가요?
도구를 생성할 때 어떤 정보가 포함되어야 하나요?

<Question
choices={[
{
text: "함수, 이름, 설명이 정의되어야 합니다.",
explain: "이 모든 것이 도구를 구성하지만, 이름과 설명은 함수와 독스트링에서 파싱될 수 있습니다.",
},
{
text: "이름만 필요합니다.",
explain: "적절한 도구 문서를 위해서는 함수와 설명/독스트링도 필요합니다.",
},
{
text: "설명만 필요합니다.",
explain: "에이전트가 도구를 선택할 때 실행할 코드가 있어야 하므로 함수가 필요합니다.",
},
{
text: "함수만 필요합니다.",
explain: "이름과 설명은 제공된 함수의 이름과 독스트링으로 기본 설정됩니다.",
correct: true
}
]}
/>

---

이 퀴즈를 마치신 것을 축하드립니다 🥳, 만약 몇 가지 요소를 놓쳤다면, 지식을 보강하기 위해 해당 챕터를 다시 읽어보세요. 통과하셨다면, 이 구성 요소들을 사용하여 더 깊이 탐구할 준비가 된 것입니다!