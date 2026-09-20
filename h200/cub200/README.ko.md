# CUB-200-2011 / H200 MIG1 실행 준비

## 범위

Lee-Wonwoo1 컨테이너에서 5모델 × seeds501/509/521 =15runs를 순차 실행한다.
사전학습 가중치를 사용하지 않는다. 기존 COCO/ADE 실행 파일·큐는 수정하지 않는다.
이 브랜치의 ImageNet 공통 helper 변경은 진행 출력 간격을 환경변수로 받는 부분뿐이다.

| 설정 | 값 |
|---|---|
| 데이터 | 공식 CUB-200-2011 train5994/test5794,200classes |
| 모델 | VA-K96,VA-K128,ConvNeXtV2-Atto,TinyViM-S,ParC-Net-S |
| 초기화 | 모든 run scratch |
| 학습 | 224×224,100epochs,AdamW LR.003/WD.05,5epoch warmup,cosine |
| 증강 | rand-m9-mstd0.5-inc1,Mixup.8,random erasing.25,label smoothing.1 |
| 메모리/속도 | BF16,fused AdamW,channels-last,비동기 CUDA 입력 복사,VA/native scan |
| 배치 | 모든 모델 물리32,누적8,목표유효256 |
| 마지막 배치 | 기존 helper처럼 drop_last 적용:5984images/epoch,마지막 누적창96images |
| update수 | 24/epoch,총2400/run |
| 평가 | epoch100의 공식 test 전체,Top1 mean±sample SD,3seeds |

bounding box/part annotation은 사용하지 않는다. test로 epoch·모델을 선택하지 않는다.
작은 데이터에서 같은 100epoch가 충분한 수렴을 보장하지는 않는다.
정규화/분류 head는 모델 고유 구조를 유지하고 출력 클래스 수만200으로 맞춘다.
따라서 ImageNet1000-class 설정보다 분류기 parameter count가 작다.

## 파일과 저장 위치

- 진입점: `bash h200/cub200/run.sh`
- 먼저 사전 점검만: `bash h200/cub200/run.sh --preflight-only`
- config: `h200/cub200/campaign.json`
- output: `/app/output/Lee-Wonwoo1/cub200-scratch-100ep-v1`
- 각 run: `runs/<model>-seed<seed>/`
- 5update마다 `step-progress.json`,epoch마다 `progress.json`/`checkpoint.pt`
- `telemetry.jsonl`, `console.log`, `supervisor.json`,최종 `result.json`
- 전체완료 후 `summary.json`에 5모델별 mean/sample SD

공식 데이터 출처: https://data.caltech.edu/records/65de6-vp158
공식 공개 MD5:97eceeb196236b17998738112f37df78.
압축 경로 이탈/링크를 거절하고 별도 staging에 해제한다. 기존 손상 데이터는 덮어쓰지 않는다.
외부 source는 고정 commit 두 개와 SHA256 검증된 CUDA12.8/Python3.13 wheel만 사용한다.
ParC/TinyViM upstream license는 기존 manifest 기준 NOASSERTION이므로 연구 용도이며,
외부 소스나 가중치를 이 저장소에 재배포하지 않는다.

## W&B와 킬 스위치

W&B project=`daehwa/alphabet2d-cub200`,group=`cub200-scratch-100ep-lee-v1`.
사전 점검과 실제15runs는 구분된 run이다. GPU를 쓰기 전에 온라인 init이 성공해야 한다.
GPU trainer와 CPU supervisor는 별도 프로세스다. W&B Stop 요청은 supervisor의
SDK가 받고 trainer에 SIGTERM을 전달한다. 정상 응답하면 현재 epoch를 마친 뒤 저장한다.
600초 내 종료하지 못하면 프로세스 그룹 전체를 SIGKILL한다. 마지막 저장 이후
진행분은 강제 종료 시 잃을 수 있다. 종료 뒤 STOP marker가 남아 다음 run을 실행하지 않는다.
30분 동안 update 변화가 없을 때도 종료한다. 사전 점검은10분으로 제한한다.
이것은 컨테이너 내부 프로세스 종료이며 Kubernetes 자원 반납을 보장하지 않는다.

로그는 로컬 파일과 W&B live file로 남기고,5초마다 heartbeat/진행량을 반영한다.
GPU 초기화/커널 컴파일 중에는 첫 update가 아직 없을 수 있다.
W&B 서버/네트워크 장애 시 원격 Stop 전달을 보장할 수 없으며 로컬 로그와 stall watchdog이 남는다.
bootstrap(환경/데이터 설치) 전에는 W&B supervisor가 아직 실행되지 않는다.
이 구간은 bootstrap.log와 다운로드/설치 timeout으로 관측·제한한다.

## H200 인증 설정

기본 실행은 `/cub-v1` 전용 W&B 릴레이를 사용한다. 실제 API key는 Worker의 기존
비공개 secret에만 남고 공개 요청에는 SDK 형식용 무권한 placeholder만 사용한다.
기존 H200 egress IP 제한과 정확한 프로젝트·17개 run ID·SDK 요청 해시를 모두 검사한다.
공유 NAT 내부의 다른 사용자를 암호학적으로 구분하는 개인 인증은 아니다.
일반적인 private runtime에서는 아래 설정도 사용할 수 있다:

1. runtime의 `WANDB_API_KEY` 또는 `WANDB_API_KEY_FILE`.
2. CUB project/run들을 허용하는 인증 릴레이와 그에 맞는 `WANDB_BASE_URL` 설정.

공개 GitHub 이슈/명령/커밋에 실제 API key를 넣지 않는다.
릴레이가 배포되지 않았거나 IP/SDK 요청이 허용되지 않으면 온라인 연결 검사에서
데이터 다운로드나 GPU 학습 전에 명시적으로 종료한다.
`python3 scripts/cub200_run_ids.py`로 연결점검1+사전점검1+실험15개의 고정 run ID를
출력할 수 있다. 릴레이 연동 시 해당17개 ID와 프로젝트에 한정해 허용해야 한다.
기본 릴레이는 attempt0만 허용한다. 새 attempt는 별도 범위 갱신이 필요하다.
H200 요청은 사용자가 제출하며 이 코드 준비 과정에서는 제출하지 않았다.

## 재개

동일 source·split·recipe·workers의 epoch checkpoint에서 optimizer와 Python/NumPy/
Torch/CUDA/DataLoader RNG를 복원한다. epoch 중간 재개/다른 worker수로의 무손실
재개를 주장하지 않는다. 계약이 바뀌면 자동 변환하지 않고 거절한다.
완료된 결과는 source/contract/모델/seed/파일 존재 검증 뒤 재학습하지 않고 건너뛴다.
Stop marker 제거는 명시적인 운영자 재개 결정 후에만 한다. W&B에서 stopped된 run을
재개할 때는 `CUB_TELEMETRY_ATTEMPT`를 증가시켜 별도 로깅 run으로 연결한다.
W&B에 저장 epoch가 있는데 checkpoint가 안 보이면 새 학습으로 덮어쓰지 않는다.
서버 볼륨이 새 job에서 안 보일 수 있으며 W&B에 모델 가중치를 자동 업로드하지는 않는다.

## 검증 근거

- 공식 split 구조/경로 검사,안전한 압축 해제 규칙,오류 전파,Stop,강제종료 테스트.
- 동일 CPU fixture에서 epoch1 저장 후 재개한 epoch2 결과가 연속학습과 정확히 일치.
- 4모델 실제224×224 CPU forward:200class shape/finite 통과.
- TinyViM:200class 생성/parameter count 확인,실제 CUDA forward는 H200 사전점검에서 수행.
- CPU-only live W&B canary에서 파일 업로드,remote stop 요청 수락,자식 종료까지 확인.
  https://wandb.ai/daehwa/alphabet2d-cub200/runs/1iqh3gf7
- H200의15runs를 실행하거나 H200 속도/메모리를 실측한 것은 아님.
