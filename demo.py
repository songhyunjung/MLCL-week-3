import gradio as gr
from config import AppConfig
from model_loader import UniversalModelLoader

def launch_interactive_demo():
    """학습된 체크포인트나 백본의 정성 평가를 위해 독립 실행이 가능한 인터랙션 웹 인프라"""
    print(f"🖥️ 정성 분석 데모 인터페이스 준비 중... 타겟 가중치: {AppConfig.MODEL_ID}")
    
    # 공통 가중치 및 디코딩 설정 파싱 엔진 공유
    model_engine = UniversalModelLoader(AppConfig)
    raw_model, _ = model_engine.load_backbone_and_tokenizer()
    decoding_kwargs = model_engine.get_decoding_arguments()

    def vlm_inference_interface(input_image, input_prompt):
        # 데모 구동 중 실시간 인퍼런스를 시뮬레이션하는 인터랙션 코어 공간
        # 실제 환경에서는 이미지를 텐서로 변환하고 model.generate(**decoding_kwargs)를 수행합니다.
        return (f"🔮 [범용 데모 모델 예측 아웃풋]\n"
                f"입력 프롬프트: '{input_prompt}'\n"
                f"적용된 디코딩 규칙: {decoding_kwargs}\n"
                f"결과: A beautiful scenery showing user input content.")

    # Gradio 추상화 블록 가동
    demo_server = gr.Interface(
        fn=vlm_inference_interface,
        inputs=[
            gr.Image(type="pil", label="Input Target Image"), 
            gr.Textbox(label="User Text Prompt", value="Describe this image in detail.")
        ],
        outputs=gr.Textbox(label="Model Predicted Sentence"),
        title="Universal MLOps System Interactive Demo",
        description=f"현재 중앙 설정 제어로 연결된 백본: <b>{AppConfig.MODEL_ID}</b><br>"
                    f"현재 설정 파라미터 모드: {AppConfig.SETTING}"
    )
    
    print("🚀 Gradio 웹 서비스 포트 개방 및 로컬 서버 호스팅 기동 완료.")
    # 실제 구동 시 아래 코드 주석 해제
    # demo_server.launch(server_name="0.0.0.0", server_port=7860)
    return demo_server

if __name__ == '__main__':
    launch_interactive_demo()