# 첫 번째 LangGraph 구축하기

이제 구성 요소를 이해했으니, 첫 번째 기능적 그래프를 구축하여 실제로 적용해 봅시다. 알프레드의 이메일 처리 시스템을 구현할 것입니다. 알프레드는 다음을 수행해야 합니다.

1. 수신 이메일 읽기
2. 스팸 또는 합법적인 이메일로 분류
3. 합법적인 이메일에 대한 예비 응답 초안 작성
4. 합법적인 경우 웨인 씨에게 정보 전송 (인쇄만)

이 예시는 LLM 기반 의사 결정을 포함하는 LangGraph로 워크플로우를 구성하는 방법을 보여줍니다. 도구가 포함되지 않았으므로 에이전트로 간주할 수는 없지만, 이 섹션은 에이전트보다는 LangGraph 프레임워크 학습에 더 중점을 둡니다.

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/langgraph/mail_sorting.ipynb" target="_blank">이 노트북</a>에서 코드를 따라 할 수 있습니다.
</Tip>

## 우리의 워크플로우

다음은 우리가 구축할 워크플로우입니다.
<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/first_graph.png" alt="First LangGraph"/>

## 환경 설정

먼저, 필요한 패키지를 설치해 봅시다.

```python
%pip install langgraph langchain_openai
```

다음으로, 필요한 모듈을 임포트해 봅시다.

```python
import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
```

## 1단계: 상태 정의

알프레드가 이메일 처리 워크플로우 중에 추적해야 하는 정보를 정의해 봅시다.

```python
class EmailState(TypedDict):
    # 처리 중인 이메일
    email: Dict[str, Any]  # 제목, 발신자, 본문 등 포함

    # 이메일 카테고리 (문의, 불만, 등)
    email_category: Optional[str]

    # 이메일이 스팸으로 표시된 이유
    spam_reason: Optional[str]

    # 분석 및 결정
    is_spam: Optional[bool]
    
    # 응답 생성
    email_draft: Optional[str]
    
    # 처리 메타데이터
    messages: List[Dict[str, Any]]  # 분석을 위한 LLM과의 대화 추적
```

> 💡 **팁:** 모든 중요한 정보를 추적할 수 있을 만큼 포괄적인 상태를 만들되, 불필요한 세부 정보로 부풀리지 마세요.

## 2단계: 노드 정의

이제 노드를 구성할 처리 함수를 만들어 봅시다.

```python
# LLM 초기화
model = ChatOpenAI(temperature=0)

def read_email(state: EmailState):
    """알프레드가 수신 이메일을 읽고 기록합니다."""
    email = state["email"]
    
    # 여기서 초기 전처리를 수행할 수 있습니다.
    print(f"알프레드가 {email['sender']}로부터 온 이메일을 처리 중입니다. 제목: {email['subject']}")
    
    # 여기서는 상태 변경이 필요 없습니다.
    return {}

def classify_email(state: EmailState):
    """알프레드가 LLM을 사용하여 이메일이 스팸인지 합법적인지 판단합니다."""
    email = state["email"]
    
    # LLM을 위한 프롬프트 준비
    prompt = f"""
    집사 알프레드로서, 이 이메일을 분석하여 스팸인지 합법적인지 판단하세요.
    
    이메일:
    보낸 사람: {email['sender']}
    제목: {email['subject']}
    본문: {email['body']}
    
    먼저, 이 이메일이 스팸인지 판단하세요. 스팸이라면 이유를 설명하세요.
    합법적이라면 분류하세요 (문의, 불만, 감사 등).
    """
    
    # LLM 호출
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    # 응답을 파싱하는 간단한 로직 (실제 앱에서는 더 견고한 파싱이 필요합니다)
    response_text = response.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text
    
    # 스팸인 경우 이유 추출
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()
    
    # 합법적인 경우 카테고리 결정
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break
    
    # 추적을 위한 메시지 업데이트
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # 상태 업데이트 반환
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }

def handle_spam(state: EmailState):
    """알프레드가 스팸 이메일을 메모와 함께 버립니다."""
    print(f"알프레드가 이메일을 스팸으로 표시했습니다. 이유: {state['spam_reason']}")
    print("이메일이 스팸 폴더로 이동되었습니다.")
    
    # 이 이메일 처리가 완료되었습니다.
    return {}

def draft_response(state: EmailState):
    """알프레드가 합법적인 이메일에 대한 예비 응답 초안을 작성합니다."""
    email = state["email"]
    category = state["email_category"] or "general"
    
    # LLM을 위한 프롬프트 준비
    prompt = f"""
    집사 알프레드로서, 이 이메일에 대한 정중한 예비 응답 초안을 작성하세요.
    
    이메일:
    보낸 사람: {email['sender']}
    제목: {email['subject']}
    본문: {email['body']}
    
    이 이메일은 다음으로 분류되었습니다: {category}
    
    Mr. Hugg가 보내기 전에 검토하고 개인화할 수 있는 간결하고 전문적인 응답 초안을 작성하세요.
    """
    
    # LLM 호출
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    # 추적을 위한 메시지 업데이트
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # 상태 업데이트 반환
    return {
        "email_draft": response.content,
        "messages": new_messages
    }

def notify_mr_hugg(state: EmailState):
    """알프레드가 Mr. Hugg에게 이메일을 알리고 응답 초안을 제시합니다."""
    email = state["email"]
    
    print("\n" + "="*50)
    print(f"선생님, {email['sender']}로부터 이메일을 받으셨습니다.")
    print(f"제목: {email['subject']}")
    print(f"카테고리: {state['email_category']}")
    print("\n검토를 위해 응답 초안을 준비했습니다:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")
    
    # 이 이메일 처리가 완료되었습니다.
    return {}
```

## 3단계: 라우팅 로직 정의

분류 후 어떤 경로를 택할지 결정하는 함수가 필요합니다.

```python
def route_email(state: EmailState) -> str:
    """스팸 분류를 기반으로 다음 단계를 결정합니다."""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"
```

> 💡 **참고:** 이 라우팅 함수는 LangGraph에 의해 호출되어 분류 노드 후에 어떤 엣지를 따라갈지 결정합니다. 반환 값은 조건부 엣지 매핑의 키 중 하나와 일치해야 합니다.

## 4단계: StateGraph 생성 및 엣지 정의

이제 모든 것을 연결해 봅시다.

```python
# 그래프 생성
email_graph = StateGraph(EmailState)

# 노드 추가
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

# 엣지 시작
email_graph.add_edge(START, "read_email")
# 엣지 추가 - 흐름 정의
email_graph.add_edge("read_email", "classify_email")

# classify_email에서 조건부 분기 추가
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response"
    }
)

# 최종 엣지 추가
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

# 그래프 컴파일
compiled_graph = email_graph.compile()
```

LangGraph에서 제공하는 특수 `END` 노드를 어떻게 사용하는지 주목하십시오. 이는 워크플로우가 완료되는 터미널 상태를 나타냅니다.

## 5단계: 애플리케이션 실행

합법적인 이메일과 스팸 이메일로 그래프를 테스트해 봅시다.

```python
# 합법적인 이메일 예시
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "서비스에 대한 질문",
    "body": "Mr. Hugg께, 동료로부터 귀하를 소개받았으며 귀하의 컨설팅 서비스에 대해 더 자세히 알고 싶습니다. 다음 주에 통화 일정을 잡을 수 있을까요? 감사합니다, John Smith"
}

# 스팸 이메일 예시
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "5,000,000달러 당첨!!!",
    "body": "축하합니다! 귀하는 국제 복권 당첨자로 선정되었습니다! 5,000,000달러 상금을 받으려면 은행 정보와 처리 수수료 100달러를 보내주십시오."
}

# 합법적인 이메일 처리
print("\n합법적인 이메일 처리 중...")
legitimate_result = compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})

# 스팸 이메일 처리
print("\n스팸 이메일 처리 중...")
spam_result = compiled_graph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})
```

## 6단계: Langfuse로 메일 분류 에이전트 검사 📡

알프레드가 메일 분류 에이전트를 미세 조정하면서, 실행 디버깅에 지쳐가고 있습니다. 에이전트는 본질적으로 예측 불가능하고 검사하기 어렵습니다. 그러나 그는 궁극적인 스팸 감지 에이전트를 구축하고 프로덕션에 배포하는 것을 목표로 하므로, 향후 모니터링 및 분석을 위한 강력한 추적 가능성이 필요합니다.

이를 위해 알프레드는 [Langfuse](https://langfuse.com/)와 같은 관찰 도구를 사용하여 에이전트를 추적하고 모니터링할 수 있습니다.

먼저, Langfuse를 pip 설치합니다:
```python
%pip install -q langfuse
```

둘째, Langchain을 pip 설치합니다 (LangFuse를 사용하므로 LangChain이 필요합니다):
```python
%pip install langchain
```

다음으로, Langfuse API 키와 호스트 주소를 환경 변수로 추가합니다. [Langfuse Cloud](https://cloud.langfuse.com)에 가입하거나 [Langfuse를 자체 호스팅](https://langfuse.com/self-hosting)하여 Langfuse 자격 증명을 얻을 수 있습니다.

```python
import os
 
# 프로젝트 설정 페이지에서 프로젝트 키 가져오기: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..." 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU 지역
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US 지역
```

그런 다음, [Langfuse `callback_handler`](https://langfuse.com/docs/integrations/langchain/tracing#add-langfuse-to-your-langchain-application)를 구성하고, 그래프 호출에 `langfuse_callback`을 추가하여 에이전트를 계측합니다: `config={"callbacks": [langfuse_handler]}`.

```   
from langfuse.langchain import CallbackHandler

# LangGraph/Langchain용 Langfuse CallbackHandler 초기화 (추적)
langfuse_handler = CallbackHandler()

# 합법적인 이메일 처리
legitimate_result = compiled_graph.invoke(
    input={"email": legitimate_email, "is_spam": None, "spam_reason": None, "email_category": None, "draft_response": None, "messages": []},
    config={"callbacks": [langfuse_handler]}
)
```

알프레드가 이제 연결되었습니다 🔌! LangGraph의 실행이 Langfuse에 기록되어 에이전트의 동작을 완전히 파악할 수 있습니다. 이 설정을 통해 그는 이전 실행을 다시 방문하고 메일 분류 에이전트를 더욱 개선할 준비가 되었습니다.

![Langfuse의 예시 추적](https://langfuse.co/images/cookbook/huggingface-agent-course/langgraph-trace-legit.png)

_[합법적인 이메일 추적에 대한 공개 링크](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/f5d6d72e-20af-4357-b232-af44c3728a7b?timestamp=2025-03-17T10%3A13%3A28.413Z&observation=6997ba69-043f-4f77-9445-700a033afba1)_

## 그래프 시각화

LangGraph는 구조를 더 잘 이해하고 디버깅하기 위해 워크플로우를 시각화할 수 있도록 합니다:

```python
compiled_graph.get_graph().draw_mermaid_png()
```
<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/mail_flow.png" alt="Mail LangGraph"/>

이는 노드가 어떻게 연결되어 있고 취할 수 있는 조건부 경로를 보여주는 시각적 표현을 생성합니다.

## 우리가 구축한 것

우리는 다음을 수행하는 완전한 이메일 처리 워크플로우를 만들었습니다:

1. 수신 이메일 수신
2. LLM을 사용하여 스팸 또는 합법적인 이메일로 분류
3. 스팸을 버림으로써 처리
4. 합법적인 이메일의 경우, 응답 초안을 작성하고 Mr. Hugg에게 알림

이는 명확하고 구조화된 흐름을 유지하면서 LLM으로 복잡한 워크플로우를 오케스트레이션하는 LangGraph의 힘을 보여줍니다.

## 핵심 요점

- **상태 관리**: 이메일 처리의 모든 측면을 추적하기 위한 포괄적인 상태를 정의했습니다.
- **노드 구현**: LLM과 상호 작용하는 기능적 노드를 만들었습니다.
- **조건부 라우팅**: 이메일 분류를 기반으로 분기 로직을 구현했습니다.
- **터미널 상태**: 워크플로우의 완료 지점을 표시하기 위해 END 노드를 사용했습니다.

## 다음 단계는?

다음 섹션에서는 워크플로우에서 인간 상호 작용을 처리하고 여러 조건을 기반으로 더 복잡한 분기 로직을 구현하는 것을 포함하여 LangGraph의 더 고급 기능을 탐색할 것입니다.