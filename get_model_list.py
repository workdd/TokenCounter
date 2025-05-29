# fetch_models.py

from huggingface_hub import HfApi

# ⚙️ 상수
DOWNLOAD_THRESHOLD = 100000  # 최소 다운로드 수
MODEL_LIMIT = 10000  # 한 번에 가져올 최대 모델 수
OUTPUT_FILE = "models.txt"


def fetch_and_save_models(
    limit: int = MODEL_LIMIT,
    min_downloads: int = DOWNLOAD_THRESHOLD,
    output_file: str = OUTPUT_FILE,
):
    """
    1) 다운로드 순으로 인기 모델 가져오기
    2) 각 모델의 tags에 'text-generation'이 포함된 것만 선별
    3) 다운로드 수(min_downloads) 이상인 것만 남기기
    4) 알파벳 순 정렬 후 파일로 저장
    """
    api = HfApi()
    # 1) 인기 순으로 모델 가져오기
    models = api.list_models(sort="downloads", direction=-1, limit=limit)
    llm_models = []
    for m in models:
        # 2) 'text-generation' 태그 수동 검사
        tags = [t.lower() for t in (m.tags or [])]
        if "text-generation" not in tags:
            continue
        # 3) 다운로드 수 필터
        if getattr(m, "downloads", 0) < min_downloads:
            continue
        llm_models.append(m.modelId)

    # 4) 알파벳 순 정렬 및 중복 제거
    llm_models = sorted(set(llm_models))

    # 5) 파일에 쓰기
    with open(output_file, "w") as f:
        for model_id in llm_models:
            f.write(model_id + "\n")

    print(f"Saved {len(llm_models)} LLM models to {output_file}")


if __name__ == "__main__":
    fetch_and_save_models()
