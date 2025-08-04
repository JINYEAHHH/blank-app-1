"""
Streamlit 통계 학습 앱 - "어떤 대푯값이 좋을까?" 페이지
OpenAI 없이 작동하는 안전한 버전 (에러 해결됨!)

필요한 패키지 설치:
pip install streamlit pandas numpy

이것만 설치하면 바로 작동합니다! 🚀
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import Counter

# 페이지 설정
st.set_page_config(
    page_title="🤔 어떤 대푯값이 좋을까?",
    page_icon="📊",
    layout="wide"
)

# 페이지 설정
st.set_page_config(
    page_title="🤔 어떤 대푯값이 좋을까?",
    page_icon="📊",
    layout="wide"
)

# CSS 스타일링
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.scenario-box {
    background: #f8f9fa;
    border: 2px solid #667eea;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.stats-display {
    background: #e6f3ff;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.data-item {
    background: #667eea;
    color: white;
    padding: 8px 12px;
    margin: 4px;
    border-radius: 20px;
    display: inline-block;
    font-weight: bold;
}

.outlier {
    background: #fc8181 !important;
}

.mode-highlight {
    background: #ffd700 !important;
    color: #333 !important;
}

.feedback-positive {
    background: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 8px;
    border-left: 5px solid #28a745;
}

.feedback-negative {
    background: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 8px;
    border-left: 5px solid #dc3545;
}

.ai-section {
    background: #f0f8ff;
    border: 2px solid #4299e1;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🤔 어떤 대푯값이 좋을까?</h1>
    <p>같은 자료라도 상황과 목적에 따라 <strong>적절한 대푯값이 달라질 수 있어요!</strong> 🎯</p>
</div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'answers_submitted' not in st.session_state:
    st.session_state.answers_submitted = {}
if 'ai_examples_checked' not in st.session_state:
    st.session_state.ai_examples_checked = {}
if 'answers_submitted' not in st.session_state:
    st.session_state.answers_submitted = {}
if 'ai_examples_checked' not in st.session_state:
    st.session_state.ai_examples_checked = {}

# 통계 계산 함수
def calculate_stats(data):
    mean = np.mean(data)
    median = np.median(data)
    mode_counter = Counter(data)
    mode = mode_counter.most_common(1)[0][0]
    return {
        'mean': round(mean, 1),
        'median': median,
        'mode': mode
    }

# 시나리오 데이터
scenarios = {
    1: {
        'title': '예제 1: 제기차기 횟수',
        'data': [4, 5, 6, 6, 6, 7, 7, 23],
        'question': '23처럼 극단적으로 큰 값이 끼어 있다면 평균, 중앙값, 최빈값 중 어떤 값이 더 적절할까요?',
        'outlier_index': 7
    },
    2: {
        'title': '예제 2: 신발 판매 사이즈',
        'data': [250, 250, 250, 260, 260, 270, 280],
        'question': '가장 많이 팔린 사이즈를 알고 싶다면 평균, 중앙값, 최빈값 중 어떤 값이 가장 유용할까요?',
        'mode_indices': [0, 1, 2]
    }
}

st.header("📊 예제 분석 — 어떤 대푯값이 적절할까?")

# 두 시나리오를 나란히 표시
col1, col2 = st.columns(2)

for i, (scenario_id, scenario) in enumerate(scenarios.items()):
    with col1 if i == 0 else col2:
        st.markdown(f"""
        <div class="scenario-box">
            <h3>{scenario['title']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 데이터 표시
        data_html = ""
        for idx, value in enumerate(scenario['data']):
            if scenario_id == 1 and idx == scenario.get('outlier_index'):
                data_html += f'<span class="data-item outlier">{value}</span> '
            elif scenario_id == 2 and idx in scenario.get('mode_indices', []):
                data_html += f'<span class="data-item mode-highlight">{value}</span> '
            else:
                data_html += f'<span class="data-item">{value}</span> '
        
        st.markdown(f"**데이터:** {data_html}", unsafe_allow_html=True)
        
        # 대푯값 계산 및 표시
        stats = calculate_stats(scenario['data'])
        
        with st.expander("📈 대푯값 보기", expanded=True):
            st.markdown(f"""
            <div class="stats-display">
                <strong>📏 평균:</strong> {stats['mean']}<br>
                <strong>📐 중앙값:</strong> {stats['median']}<br>
                <strong>🎯 최빈값:</strong> {stats['mode']}
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"❓ {scenario['question']}")
        
        # 답변 입력
        st.subheader("💭 당신의 답변")
        
        best_stat = st.selectbox(
            "가장 적절한 대푯값:",
            ["선택하세요", "평균", "중앙값", "최빈값"],
            key=f"stat_select_{scenario_id}"
        )
        
        reason = st.text_area(
            "이유를 설명해주세요:",
            placeholder="왜 이 대푯값이 가장 적절한지 설명해보세요...",
            key=f"reason_{scenario_id}",
            height=100
        )
        
        # 답변 제출 버튼
        if st.button(f"답변 제출", key=f"submit_{scenario_id}"):
            if best_stat != "선택하세요" and reason.strip():
                with st.spinner("🤖 AI가 답변을 분석하고 있습니다..."):
                    # 시뮬레이션 로딩
                    time.sleep(2)
                    
                    # 정답 체크 로직 (키워드 기반)
                    correct_answers = {
                        1: {"stats": ["중앙값", "최빈값"], "keywords": ["극단값", "이상치", "왜곡", "영향", "극단"]},
                        2: {"stats": ["최빈값"], "keywords": ["많이 팔린", "빈도", "자주", "흔한"]}
                    }
                    
                    correct = correct_answers[scenario_id]
                    is_correct_stat = best_stat in correct["stats"]
                    has_key_reason = any(keyword in reason.lower() for keyword in correct["keywords"])
                    
                    if is_correct_stat and has_key_reason:
                        st.markdown("""
                        <div class="feedback-positive">
                            ✅ <strong>훌륭합니다!</strong><br>
                            상황에 가장 적절한 대푯값을 선택하고 타당한 이유를 제시했습니다.
                        </div>
                        """, unsafe_allow_html=True)
                    elif is_correct_stat:
                        st.markdown("""
                        <div class="feedback-negative">
                            💭 <strong>대푯값 선택은 맞지만</strong><br>
                            이유를 더 구체적으로 설명해보세요. 자료의 특성을 고려해보세요!
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="feedback-negative">
                            🤔 <strong>다시 생각해보세요!</strong><br>
                            자료의 분포와 특성을 고려하여 가장 적절한 대푯값을 선택해보세요.
                        </div>
                        """, unsafe_allow_html=True)
                
                st.session_state.answers_submitted[scenario_id] = True
            else:
                st.error("❌ 대푯값과 이유를 모두 입력해주세요!")

# 구분선
st.markdown("---")

# AI 가이드 섹션
st.header("✅ 어떤 대푯값을 쓰면 좋을까?")

# 각 대푯값별 예시 입력 섹션
stat_types = {
    'mean': {'name': '평균', 'emoji': '📏', 'color': '#667eea'},
    'median': {'name': '중앙값', 'emoji': '📐', 'color': '#38b2ac'},
    'mode': {'name': '최빈값', 'emoji': '🎯', 'color': '#f6ad55'}
}

for stat_key, stat_info in stat_types.items():
    st.markdown(f"""
    <div class="ai-section">
        <h3>{stat_info['emoji']} {stat_info['name']}은 어떤 상황에서 쓰면 좋을까?</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        example_text = st.text_area(
            f"{stat_info['name']}을 사용하는 것이 좋다고 생각하는 구체적인 상황이나 예시를 적어보세요:",
            placeholder=f"예: {stat_info['name']}을 사용하면 좋은 상황을 설명해보세요...",
            key=f"{stat_key}_example",
            height=120
        )
    
    with col_right:
        if st.button(f"🤖 AI가 확인해보기", key=f"check_{stat_key}"):
            if example_text.strip():
                with st.spinner("🤖 AI가 분석하고 있습니다..."):
                    # 시뮬레이션 로딩
                    time.sleep(1.5)
                    
                    # 키워드 기반 분석 로직
                    analysis_rules = {
                        'mean': {
                            'good_keywords': ['고르게', '균등', '일정', '비슷', '평균적', '고루', '분포'],
                            'bad_keywords': ['극단', '이상치', '튀는', '치우쳐', '빈도']
                        },
                        'median': {
                            'good_keywords': ['극단', '이상치', '튀는', '치우쳐', '왜곡', '한쪽으로'],
                            'bad_keywords': ['고르게', '균등', '평균적', '빈도']
                        },
                        'mode': {
                            'good_keywords': ['많이', '자주', '흔한', '인기', '빈도', '판매량', '최다'],
                            'bad_keywords': ['평균', '중간', '균등', '고르게']
                        }
                    }
                    
                    rule = analysis_rules[stat_key]
                    lower_text = example_text.lower()
                    
                    has_good = any(keyword in lower_text for keyword in rule['good_keywords'])
                    has_bad = any(keyword in lower_text for keyword in rule['bad_keywords'])
                    
                    if has_good and not has_bad:
                        st.success(f"✅ 훌륭합니다! {stat_info['name']}이 적절한 상황을 잘 파악했습니다.")
                    elif has_good:
                        st.info(f"👍 좋습니다! {stat_info['name']}의 특성을 잘 이해하고 있어요.")
                    else:
                        st.warning(f"💡 {stat_info['name']}이 왜 적절한지 더 구체적으로 설명해보세요!")
                
                st.session_state.ai_examples_checked[stat_key] = True
            else:
                st.error("❌ 예시를 입력해주세요!")

# 결론 섹션
st.markdown("---")
st.markdown("""
<div class="main-header">
    <h2>🎯 핵심 포인트!</h2>
    <p>💡 <strong>완벽한 대푯값은 없어요!</strong><br>
    상황과 목적에 맞는 <strong>가장 적절한 대푯값</strong>을 선택하는 것이 중요합니다! ✨</p>
</div>
""", unsafe_allow_html=True)

# 진도 체크
completed_activities = 0
total_activities = 5  # 2개 시나리오 + 3개 AI 예시

completed_activities += len(st.session_state.answers_submitted)
completed_activities += len(st.session_state.ai_examples_checked)

progress_bar = st.progress(completed_activities / total_activities)
st.write(f"진도율: {completed_activities}/{total_activities} 완료 ({int(completed_activities/total_activities*100)}%)")

if completed_activities == total_activities:
    st.balloons()
    st.success("🎉 모든 활동을 완료했습니다! 다음 단계로 넘어가세요!")