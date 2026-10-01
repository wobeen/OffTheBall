# 제3자 고지

조사 기준일: 2026-10-01

이 문서는 OffTheBall이 설계 참고 또는 실행 의존성으로 쓰는 외부 프로젝트, 모델, 데이터의 출처와 라이선스를 기록한다. 법률 자문이 아니며, 배포 전에 각 원문을 다시 확인해야 한다.

## 현재 코드에 포함·의존하는 항목

| 항목 | 용도 | 라이선스 | 비고 |
|---|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) `8.4.152` (`requirements.lock.txt`) | YOLO 추론, ByteTrack 추적 | AGPL-3.0 (기업 라이선스 별도 제공) | 의존성으로 설치해 사용한다. 이 프로젝트를 외부에 배포하거나 네트워크 서비스로 제공하기 전에 AGPL 의무와 프로젝트 라이선스 방침을 결정해야 한다. |
| `models/yolo11n.pt` | 사람 탐지 기본 모델 | Ultralytics 공식 가중치. AGPL-3.0 적용으로 알려져 있음 | 이 저장소에서 이 파일의 배포 가능 여부는 별도로 확인하지 않았다. |
| 나머지 직접 의존성 (`numpy`, `opencv-python`, `mss`, `Pillow`, `requests`, `lap`, `scipy`) | 입력, 영상, 추적 보조 | 미조사 | `pyproject.toml` 참고. 배포 전에 조사한다. |

이 프로젝트 자체의 코드는 [LICENSE](LICENSE)의 MIT 라이선스를 따른다(2026-10-01 결정). 단, 위 Ultralytics는 AGPL-3.0이며 MIT가 이를 바꾸지 않는다. Ultralytics를 포함해 배포하거나 네트워크 서비스로 제공할 때는 결합물에 AGPL 의무가 적용될 수 있으므로 그 시점에 별도로 검토한다.

라이선스는 저작권자가 이후 버전부터 바꿀 수 있다. 이미 MIT로 배포한 버전은 받은 사람이 계속 MIT로 쓸 수 있다. 외부 기여를 받기 시작하면 기여자 동의 없이는 바꾸기 어려워지므로, 변경 가능성을 유지하려면 기여를 받기 전에 방침을 정한다.

## 설계 참고 대상 (코드는 가져오지 않음)

아래 저장소는 구조와 알고리즘 아이디어를 참고했다. 현재까지 외부 소스 코드를 복사하지 않았고 OpenCV 등 공개 API로 독립 구현했다. 코드를 가져오기로 결정하면 이 표에 원본 경로, 커밋, 수정 내용을 추가한다.

| 저장소 | 라이선스 | 조사한 기준 커밋 (기본 브랜치) | 참고 범위 |
|---|---|---|---|
| [Simo-03/football-player-detection](https://github.com/Simo-03/football-player-detection) | MIT | `c0c305d` (2026-07-08, `main`) | 32개 경기장 키포인트, RANSAC 보정 검증, 이전 보정 제한적 재사용, BoT-SORT 설정, 공 검출, 점유·패스 구조 |
| [roboflow/sports](https://github.com/roboflow/sports) | MIT | `42c80c0` (2025-05-27, `main`) | 표준 경기장 좌표 모델, 키포인트-경기장 대응, 전술 레이더, 팀 분류 흐름 |
| [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate) | **GPL-3.0** | `1c95834` (2026-05-02, `main`) | 경기 상태 표현, 추적 평가 지표 설계(GS-HOTA/HOTA) |
| [BlazeWild/MatchVision-AI-Sports-Video-Analytics-Tracking-Pipeline](https://github.com/BlazeWild/MatchVision-AI-Sports-Video-Analytics-Tracking-Pipeline) | MIT | `9876ed6` (2026-06-18, `main`) | 키프레임 사이 광학 흐름, 불가능한 이동 제거, 보정 진단 |
| [mguti97/PnLCalib](https://github.com/mguti97/PnLCalib) | **GPL-2.0** | `8c87391` (2026-03-17, `main`) | 점·선 기반 카메라 보정 정확도 비교 |

라이선스와 커밋은 GitHub API로 확인했다. 커밋은 조사 시점의 기본 브랜치 최신 값이며, 코드를 도입하는 시점에 다시 고정해야 한다.

2026-10-01 결정: Simo-03·roboflow/sports는 아이디어 참고만 하고, sn-gamestate·PnLCalib·MatchVision AI는 개발 계획에서 제외했다(아래 정보는 조사 기록으로 남김). 자세한 방침은 [docs/개발_계획.md](docs/개발_계획.md) 2절.

### 사용 규칙

- **MIT 저장소**: 코드를 가져오면 원본 저작권 고지와 MIT 라이선스 전문을 함께 남긴다. 가져온 파일은 원본 경로와 커밋을 이 문서에 기록한다.
- **GPL-3.0 (sn-gamestate), GPL-2.0 (PnLCalib)**: 소스 코드를 이 프로젝트에 복사하거나 링크하지 않는다. 지표 정의와 논문·문서 수준의 아이디어 참고, 그리고 별도 프로세스로 돌린 성능 비교 실험까지만 허용한다. 배포물에 포함하는 결정은 라이선스 검토 후 별도로 한다.
- MIT 표시가 상위 저장소에만 해당하는 경우가 있다. 모델 가중치, 학습 데이터셋, 하위 포함 코드의 라이선스는 항목별로 따로 확인한다.

## 모델 가중치와 데이터셋 (도입 전 확인 필요)

실제 키포인트·축구 탐지 가중치를 연결하기 전에 아래를 확인해야 한다. 지금은 어떤 가중치도 이 저장소에 포함하지 않았다.

| 항목 | 확인된 사실 | 미확인 사항 |
|---|---|---|
| Simo-03 릴리스 `v1.0.0` 가중치 (`pitch_kpts32_y8s_640_e500_AO.pt`, `best_players_gk_ball_960_s_e502.pt`, `ball_tracking_1280_e300.pt`) | 저장소는 MIT. 가중치는 YOLOv8s-pose·YOLO11s 기반이라 Ultralytics(AGPL-3.0) 영향을 받을 수 있음. 릴리스 설명은 가중치 파일명을 `player_detector.pt` 등으로 적고 있어 실제 자산 이름과 다름 | 가중치 단독 라이선스 문구, 학습 데이터 출처와 데이터 라이선스, 재배포 허용 여부 |
| roboflow/sports 예제 가중치와 [Roboflow Universe 데이터셋](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) (선수·공·경기장 키포인트) | 저장소는 MIT. 예제 README는 Ultralytics 부분을 AGPL-3.0, Supervision을 MIT로 안내 | 각 데이터셋의 개별 라이선스(Universe 데이터셋은 항목마다 다름), 가중치 배포 방식 |
| 이 프로젝트의 평가 영상 `manvsmci.mp4`와 앞으로 쓸 방송·전술캠 클립 | 사용자가 제공한 영상. 방송 중계는 저작권 대상일 수 있고, 쿠팡플레이 등 보호 콘텐츠는 캡처가 차단됨 | 출처, 용도, 재배포 가능 여부. 영상과 정답 파일은 Git에 넣지 않는다. |

## 갱신 규칙

- 외부 코드, 모델, 데이터를 도입하는 커밋에서 이 문서를 같은 커밋에 갱신한다.
- 도입 시 기록할 항목: 소스 URL, 고정한 커밋 또는 릴리스 태그, 라이선스 원문 위치, 수정 내용, 가져온 파일 경로.
