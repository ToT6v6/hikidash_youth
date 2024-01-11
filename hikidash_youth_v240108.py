
# !pip install streamlit-option-menu
# !pip install streamlit_chat
# !pip install -U sentence-transformers


# 라이브러리 불러오기
from st_aggrid import AgGrid
import webbrowser
import pandas as pd
from datetime import datetime, time
import streamlit as st
import numpy as np
from urllib.parse import quote
import ssl
from urllib.request import urlopen

from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import io
import os

# -------------------- ▼ Streamlit 웹 화면 구성 START ▼ --------------------
# -------------------- ▼ Streamlit 웹 화면 구성 START ▼ --------------------

# 레이아웃 구성하기 
st.set_page_config(
    page_title="함께 해요! 어른이집",
    page_icon="🐣",
    layout="wide",
    initial_sidebar_state="expanded"
)

    

# ------------------------------ ▼ 챗봇 구성 ▼ ------------------------------------
# Define a function to save the conversation to a CSV file 함수 - 사용자와 챗봇 간의 대화를 저장하는 함수
@st.cache_data()
def save_conversation(user_text, bot_text, filepath='conversation_history.csv'):
    # Check if the file exists
    if os.path.exists(filepath):
        mode = 'a' # append if already exists
        header = False
    else:
        mode = 'w' # make a new file if not
        header = True
    
    # Define the data to be saved
    data_to_save = {
        'user': [user_text],
        'bot': [bot_text],
    }
    
    # Create a DataFrame and append it to the CSV
    pd.DataFrame(data_to_save).to_csv(filepath, mode=mode, header=header, index=False)

# @st.cache_data() # 모델과 데이터셋을 캐싱하여 성능을 향상시킵니다.
# def cached_model():
#     model = SentenceTransformer('jhgan/ko-sroberta-multitask')
#     return model

# @st.cache_data() # 모델과 데이터셋을 캐싱하여 성능을 향상시킵니다.
# def get_dataset():
#     df = pd.read_csv('wellness_dataset.csv')
#     df['embedding'] = df['embedding'].apply(json.loads)
#     return df

# model = cached_model()
# df = get_dataset()

# -----------------------------------------------------------------------------

    ## -------------------------------------------------------------------------------------
    
# 멀티 페이지
# page1 내용물 구성하기 
def main_page():
    
    st.sidebar.header('어서오세요!')
    st.sidebar.markdown('##### 몽땅 챗봇과의 대화를 시작으로 우리 함께 가보자구요!')
    
    # 제목 넣기
    st.header('몽땅 Talk! Talk!')
    
    is_chatbot_active = True
    if is_chatbot_active:
        # 챗봇
        @st.cache_data()
        def cached_model():
            model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            return model

        @st.cache_data()
        def get_dataset():
            df = pd.read_csv('빅프임2.csv')
            df['embedding'] = df['embedding'].apply(json.loads)
            return df

        model = cached_model()
        df = get_dataset()

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = st.chat_input("이곳에 답변을 입력해주세요.")
    
    if user_input:
        embedding = model.encode(user_input)

        df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        answer = df.loc[df['distance'].idxmax()]

        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer['챗봇'])
        
        # 대화내용 저장
        save_conversation(user_input, answer['챗봇'])

    for i in range(len(st.session_state['past'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        if len(st.session_state['generated']) > i:
            message(st.session_state['generated'][i], key=str(i) + '_bot')
            


# '오늘의 미션'을 결정하는 함수
def get_todays_mission(conversations):
    # 특정 키워드에 기반한 미션을 설정
    keywords_to_missions = {
        '힘들어': '오늘은 30분정도 나가서 산책하며 하늘 사진 찍어오기!',
        '지쳤어': '오늘은 책을 읽고 인증 사진 찍어 올리기!',
        # 여기에 더 많은 키워드와 미션을 추가할 수 있습니다.
    }
    for keyword, mission in keywords_to_missions.items():
        if keyword in conversations:
            return mission
    return "오늘의 미션을 찾을 수 없습니다."

# page2 내용 구성하기
def create_schedule():
    events = [
        {"Time": "09:00", "Event": "3회차 - 마음주의 상담센터 - 강남점", "Duration": "09:00-10:00"},
        {"Time": "11:00", "Event": "1회차 - 청년모임센터파크 - 강남점", "Duration": "11:00-13:00"},
        # ... 추가 이벤트
    ]
    
    # 이벤트 데이터를 DataFrame으로 변환
    schedule_df = pd.DataFrame(events)
    return schedule_df

def page2_tab2():
    st.header("신청현황 및 완료현황")
    categories = {
        "일자리 및 창업": [
            {"title": "서울시 일자리 카페 참여", "location": "강남구", "datetime": "2023-05-20 14:00", "status": "완료"},
            {"title": "취업날개 서비스(정장대여)", "location": "강남구", "datetime": "2023-07-12 10:00", "status": "완료"},
            {"title": "미래 청년 일자리 사업 참여", "location": "강남구", "datetime": "2023-08-02 09:00", "status": "완료"},
            {"title": "서울형 청년인턴 직무캠프", "location": "강남구", "datetime": "2024-01-03 09:00", "status": "진행중"},
            {"title": "취업날개 서비스(정장대여) 2회", "location": "강남구", "datetime": "2024-01-22 15:00", "status": "예정"}
        ],
        "주거": [
            {"title": "청년 주거 안정 프로그램", "location": "서초구", "datetime": "2023-06-15 10:00", "status": "완료"},
            {"title": "서울시 청년 임대주택", "location": "강남구", "datetime": "2023-09-01 09:00", "status": "완료"},
        ],
        "진로 및 교육": [
            {"title": "진로 설계 워크샵", "location": "강남구", "datetime": "2023-10-12 13:00", "status": "완료"},
            {"title": "온라인 교육 플랫폼 지원", "location": "강남구", "datetime": "2023-11-05 15:00", "status": "완료"},
        ],
        "금융 및 생활지원": [
            {"title": "금융 상담 서비스", "location": "강남구", "datetime": "2023-12-21 11:00", "status": "완료"},
            {"title": "생활비 지원 프로그램", "location": "강남구", "datetime": "2024-01-03 10:00", "status": "진행중"},
        ],
        "마음 및 신체 건강": [
            {"title": "마음건강 세미나", "location": "강남구", "datetime": "2024-01-02 16:00", "status": "진행중"},
            {"title": "신체 건강 캠페인", "location": "강남구", "datetime": "2024-02-15 14:00", "status": "예정"},
        ],
        "문화/예술": [
            {"title": "예술 작품 전시회", "location": "강남구", "datetime": "2023-09-18 10:00", "status": "완료"},
            {"title": "지역 문화 행사 참여", "location": "강남구", "datetime": "2024-01-13 12:00", "status": "예정"}
        ],
    }

    # 각 카테고리별로 expander를 만들고 내용을 채워 넣습니다.
    for category, items in categories.items():
        with st.expander(category):
            for item in items:
                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 2, 1, 1, 1])
                with col1:
                    st.write(f"{item['title']}")
                with col2:
                    st.write(f"지역: {item['location']}")
                with col3:
                    st.write(f"날짜: {item['datetime']}")
                with col4:
                    st.checkbox("완료", value=item['status']=="완료", key=f"{item['title']}_완료")
                with col5:
                    st.checkbox("진행중", value=item['status']=="진행중", key=f"{item['title']}_진행중")
                with col6:
                    st.checkbox("예정", value=item['status']=="예정", key=f"{item['title']}_예정")


                
def page2():
    is_chatbot_active = False
    
    st.sidebar.header('나의 기록')
    st.sidebar.markdown('##### 그동안 어른이집에서 나는 무엇을 하며 성장했을까?')
    
    # 제목 넣기
    st.markdown("## 나의 기록")
    
    # 탭 구성하기
    tabs = st.tabs(['오늘의 나는?', '참여현황'])

    # '오늘의 나는?' 탭
    with tabs[0]:
        # 대화 내역 불러오기
        if os.path.exists('conversation_history.csv'):
            df = pd.read_csv('conversation_history.csv')
            conversations = ' '.join(df['user'].dropna().tolist())
            todays_mission = get_todays_mission(conversations)
        else:
            todays_mission = "대화 내역이 없습니다."
        
        cols = st.columns([2, 1])  # 왼쪽 칼럼을 오른쪽 칼럼보다 크게 설정

        with cols[0]:
            with st.container(border=True):
                st.subheader("오늘의 미션")
                st.write(todays_mission)  # 오늘의 미션 내용을 여기에 표시

            with st.container(border=True):
                # 체크리스트 항목들 정의
                daily_tasks = [
                    "아침에 일어나서 물 한 컵 마시기",
                    "아침밥 먹기",
                    "30분 이상 책 보기 혹은 공부하기",
                    "30분 이상 밖에 나가기",
                    "밤 12시 전에 취침하기"
                ]
                
                # 체크리스트 섹션
                st.subheader("오늘의 체크리스트")
                # 각 항목에 대해 체크박스 생성
                for task in daily_tasks:
                    st.checkbox(task)

            with st.container(border=True):
            # 일정표 및 완료 체크박스 표시
                st.subheader("오늘의 일정")
                schedule_df = create_schedule()
                # 일정표와 완료 체크박스를 표시
                for idx, row in schedule_df.iterrows():
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col1:
                        st.write(row['Time'])
                    with col2:
                        st.write(row['Event'])
                    with col3:
                        completed = st.checkbox("완료", key=idx)
                        if completed:
                            st.write("완료되었습니다!")
       

        with cols[1]:
            with st.container():    
                # 미션 인증 사진 올리기 섹션
                uploaded_file = st.file_uploader("미션 인증 사진 올리기", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    # 인증 사진 표시
                    image = Image.open(uploaded_file)
                    cols = st.columns([1,4])
                    with cols[0]:
                        st.image(image, caption='', width=50)
                    with cols[1]:
                        st.success('인증완료')

                    # 게이지가 7 미만일 때만 증가
                    if st.session_state.gauge < 7:
                        st.session_state.gauge += 1
            with st.container(border=True):
                # 포켓몬 이미지 표시
                if 'gauge' not in st.session_state:
                    st.session_state.gauge = 0  # 게이지 초기값 설정
                # 포켓몬 진화 상태에 따른 이미지 표시
                if st.session_state.gauge <= 2:
                    st.image('피츄.png', width=400)
                elif 3 <= st.session_state.gauge <= 5:
                    st.image('피카츄.png', width=400)
                else:
                    st.image('라이츄.png', width=400)
                # 게이지 업데이트
                gauge_display = "🔵" * st.session_state.gauge + "⚪" * (7 - st.session_state.gauge)
                st.markdown(f"###### 진화 게이지: {gauge_display}")
    
    # '그동안의 기록' 탭
    with tabs[1]:
        page2_tab2()

        
# page3 내용 구성하기
def page3():
    is_chatbot_active = False
    
    st.sidebar.header('정책 찾기')
    st.sidebar.markdown('##### 나에게 맞는 정책과 정보를 찾아볼게요.')
    # 제목 넣기
    st.markdown('## 정책 찾기')
    cols = st.columns(5)
    with cols[0]:
        # 카테고리와 관련된 웹사이트 URL을 매핑
        categories = {
            "일자리 및 창업": "https://youth.seoul.go.kr/youthConts.do?key=2310100011&sc_pbancSeCd=003&sc_bbsStngSn=2212200001&sc_bbsCtgrySn=2310200004&sc_qnaCtgryCd=&sc_faqCtgryCd=011",
            "주거": "https://youth.seoul.go.kr/content.do?key=2310100033",
            "진로 및 교육": "https://youth.seoul.go.kr/content.do?key=2310200005",
            "금융 및 생활지원": "https://youth.seoul.go.kr/youthConts.do?key=2310100061&sc_pbancSeCd=012&sc_bbsStngSn=2212200001&sc_bbsCtgrySn=2310200011&sc_qnaCtgryCd=004&sc_faqCtgryCd=005",
            "마음 및 신체 건강": "https://youth.seoul.go.kr/youthConts.do?key=2310100076&sc_pbancSeCd=009&sc_bbsStngSn=2212200001&sc_bbsCtgrySn=2310200008&sc_qnaCtgryCd=&sc_faqCtgryCd=010",
            "문화/예술": "https://youth.seoul.go.kr/youthConts.do?key=2310200023&sc_pbancSeCd=005&sc_bbsStngSn=2212200001&sc_bbsCtgrySn=2310200005&sc_qnaCtgryCd=&sc_faqCtgryCd=002",
        }
    
        # 각 카테고리에 대한 탭 생성
        for category, url in categories.items():
            st.link_button(category, url, use_container_width=True)
        
    ## -------------------------------------------------------------------------------------------
    

    ## -------------------- ▼ 사이드 바를 구성해서 페이지 연결하기 ▼ -------------------- #

# 선택한 페이지 함수에 대한 딕셔너리 생성
page_functions = {'몽땅 톡톡': main_page, '나의 기록': page2, '정책을 찾아서': page3}

# "어른이집" 페이지 선택
with st.sidebar:
    choose = option_menu("어른이집🐣", ["몽땅 톡톡", "나의 기록", "정책을 찾아서"],
                         icons=['bi bi-balloon-heart-fill', 'bi bi-balloon-heart', 'bi bi-box2-heart'], # 아이콘 변경: https://icons.getbootstrap.com/
                         menu_icon="bi bi-emoji-smile", default_index=0,
                         styles={"container": {"padding": "5!important", "background-color": "#fafafa"},
                                 "icon": {"color": "#2a2415", "font-size": "25px"}, 
                                 "nav-link": {"color": "#2a2415", "font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                 "nav-link-selected": {"background-color": "#ffd851"},
                                }
                        )

# 선택한 페이지 함수 실행
page_functions[choose]()
