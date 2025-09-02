# DVC 운영 가이드 (Cloudflare R2 원격)

## 개요
- 대용량 아티팩트(규칙 스냅샷, FAISS 인덱스)를 DVC로 관리합니다.
- 원격 스토리지는 Cloudflare R2(S3 호환)를 사용합니다.

## 요구사항
- Python 가상환경(uv 권장)
- dvc, dvc-s3 플러그인

## 초기 설정
1) 가상환경 및 의존성
```bash
uv venv .venv && . .venv/bin/activate
uv pip install dvc dvc-s3
```

2) 자격증명(권장: 환경변수)
```bash
export AWS_ACCESS_KEY_ID=<YOUR_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<YOUR_SECRET>
# 선택: export AWS_DEFAULT_REGION=auto
```

3) 원격 구성 확인
```bash
dvc remote list
cat .dvc/config
```

## 원격(R2) 구성 변경(예시)
```bash
dvc remote modify r2 url s3://<bucket>/<prefix>
dvc remote modify r2 endpointurl https://<accountid>.r2.cloudflarestorage.com
dvc remote default r2
```

## 기본 워크플로우
- 데이터 추가/갱신 후
```bash
dvc add <path>
# 필요 시 파이프라인 실행
# dvc repro

# 메타데이터 커밋
git add *.dvc dvc.yaml dvc.lock .gitignore
git commit -m "chore(dvc): update artifacts"

# 원격 업로드
dvc push
```

- 새 환경에서 데이터 받기
```bash
uv venv .venv && . .venv/bin/activate
uv pip install dvc dvc-s3

# 전체 동기화
dvc pull
# 특정 항목만
dvc pull faiss_all_minilm_l6_v2.dvc
```

## 관리 대상
- 규칙 스냅샷: artifacts/rules_snapshot.tar.gz (DVC 스테이지: rules_snapshot)
- FAISS 인덱스: faiss_* 디렉터리(각각 .dvc 메타로 추적)

## 유용한 옵션
```bash
dvc config core.autostage true   # DVC가 변경한 파일 자동 git add
dvc install                      # git 훅 설치(post-checkout/merge 후 dvc pull)
```

## 보안 권장사항
- 자격증명은 커밋하지 말고 환경변수 또는 .dvc/config.local 사용
```bash
dvc remote modify --local r2 access_key_id <KEY_ID>
dvc remote modify --local r2 secret_access_key <SECRET>
```
- 키가 외부에 노출되면 즉시 로테이션하세요.

## 상태/동기화 점검
```bash
dvc status -r r2   # 원격과 차이 확인
dvc push           # 업로드
dvc pull           # 다운로드
```

## FAQ
- Git 커밋을 DVC가 가로채나요? 아니요. 메타파일만 Git에 반영됩니다.
- 원격 변경이 자동 반영되나요? 로컬에서 dvc pull을 실행해야 반영됩니다.
