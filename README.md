# RBY1 Isaac Lab Extension

Rainbow Robotics RBY1 humanoid robot simulation environment for [Isaac Lab](https://isaac-sim.github.io/IsaacLab).

RBY1-M은 메카넘 휠 기반 모바일 휴머노이드 로봇으로, 전방향 이동과 상체 조작이 가능합니다.

<table>
<tr>
<td align="center"><b>Short Demo</b></td>
<td align="center"><b>Long Demo</b></td>
</tr>
<tr>
<td>

https://github.com/user-attachments/assets/57b017d6-ced0-4d49-a6eb-d914f58eec65

</td>
<td>

https://github.com/user-attachments/assets/d0cbd8a0-30f5-41f1-bebe-37129234b7a0

</td>
</tr>
</table>

## Robot Specification

| Component | DOF | Control |
|-----------|-----|---------|
| Mecanum Wheels | 4 (fl, fr, rl, rr) | Velocity |
| Torso | 6 (torso_0~5) | Position |
| Arms | 14 (7 per arm) | Position |
| Head | 2 (head_0~1) | Position |
| Grippers | 4 (2 per hand) | Position |
| **Total** | **30 DOF** | |

## Installation

### 1. Isaac Lab 설치

[Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)를 참고하여 설치합니다. conda 또는 uv 설치를 권장합니다.

### 2. 로봇 모델 다운로드

> **RBY1 URDF/메쉬 파일은 이 저장소에 포함되어 있지 않습니다.**
> [rby1-sdk GitHub](https://github.com/RainbowRobotics/rby1-sdk)에서 로봇 모델 파일을 직접 다운로드하여 아래 경로에 배치하세요.

```
source/rby1/rby1/assets/
├── rby1m/          # RBY1 Model M (mecanum wheel) - 주로 사용
│   ├── model.urdf
│   └── meshes/
├── rby1a/          # RBY1 Model A
└── rby1ub/         # RBY1 Upper Body
```

### 3. 패키지 설치

```bash
python -m pip install -e source/rby1
```

### 4. 설치 확인

```bash
python scripts/list_envs.py
```

---

## Tasks

### Template-Rby1-Direct-v0 : Whole-Body Control (Direct)

전체 관절(휠 4 + 상체 20)을 직접 제어하는 환경입니다. 로봇이 기본 자세를 유지하도록 학습합니다.

| Item | Detail |
|------|--------|
| Action | 24 DOF (4 wheels velocity + 20 upper body delta position) |
| Observation | 51 (joint pos 24 + joint vel 24 + projected gravity 3) |
| Episode | 10초 (1200 steps @ 120Hz, decimation 4) |
| Reward | alive bonus, joint deviation penalty, velocity/acceleration/action rate penalty |
| Termination | torso tilt > 1.0 rad 또는 timeout |

```bash
python scripts/random_agent.py --task=Template-Rby1-Direct-v0
```

---

### Template-Rby1-Nav-v0 : Navigation (Direct)

메카넘 휠만 제어하여 랜덤 목표 지점까지 이동하는 네비게이션 환경입니다. Teleop 모드에 최적화되어 있습니다.

| Item | Detail |
|------|--------|
| Action | 4 DOF (mecanum wheel velocities) |
| Observation | 9 (relative target 2 + heading 1 + wheel vel 2 + base vel 2 + distance 1 + heading error 1) |
| Episode | 300초 (teleop용 긴 에피소드) |
| Reward | distance reward, heading reward, target reached bonus (+10), action rate penalty |
| Termination | 낙하 또는 timeout |

```bash
python scripts/random_agent.py --task=Template-Rby1-Nav-v0
```

---

### Template-Rby1-v0 : Pose Control (Manager-Based)

Isaac Lab의 Manager 시스템을 사용한 상체 자세 제어 환경입니다. 휠은 제어하지 않고 상체 20 관절만 제어합니다.

| Item | Detail |
|------|--------|
| Action | 20 DOF (upper body delta position, scale 0.5 rad) |
| Observation | joint_pos_rel + joint_vel_rel (auto-managed) |
| Episode | 10초 |
| Reward | alive bonus, termination penalty, joint position L2, joint velocity L1 |
| Termination | timeout |

```bash
python scripts/random_agent.py --task=Template-Rby1-v0
```

---

### Template-Rby1-Marl-Direct-v0 : Multi-Agent RL (Direct)

Cart-Double Pendulum 기반의 멀티 에이전트 RL 벤치마크 환경입니다. (RBY1 로봇 환경이 아닌 MARL 레퍼런스 구현)

| Item | Detail |
|------|--------|
| Agents | 2 (cart + pendulum) |
| Action | cart: 1 (force), pendulum: 1 (torque) |
| Episode | 5초 |

```bash
python scripts/random_agent.py --task=Template-Rby1-Marl-Direct-v0
```

---

## Teleop Mode (Keyboard)

키보드로 RBY1의 메카넘 휠을 직접 조종해볼 수 있습니다.

### 실행

```bash
# 기본 실행
python scripts/teleop_keyboard.py --task=Template-Rby1-Nav-v0 --num_envs=1

# 동영상 녹화 포함
python scripts/teleop_keyboard.py --task=Template-Rby1-Nav-v0 --num_envs=1 --video --video_length=500
```

녹화된 동영상은 `./videos/` 디렉토리에 저장됩니다.

### 조작법

| Key | Action |
|-----|--------|
| Arrow Up / Down | 전진 / 후진 |
| Arrow Left / Right | 좌측 / 우측 횡이동 (strafe) |
| Z / X | 반시계 / 시계 방향 회전 |
| L | 정지 |

### Mecanum Wheel Kinematics

```
wheel_fl = vx - vy - wz
wheel_fr = vx + vy + wz
wheel_rl = vx + vy - wz
wheel_rr = vx - vy + wz
```

---

## Training

4가지 RL 프레임워크를 지원합니다:

```bash
# RSL-RL
python scripts/rsl_rl/train.py --task=<TASK_NAME>

# Stable Baselines 3
python scripts/sb3/train.py --task=<TASK_NAME>

# SKRL
python scripts/skrl/train.py --task=<TASK_NAME>

# RL-Games
python scripts/rl_games/train.py --task=<TASK_NAME>
```

---

## Project Structure

```
isaaclab-rby1/
├── scripts/
│   ├── teleop_keyboard.py          # 키보드 텔레오퍼레이션
│   ├── random_agent.py             # 랜덤 에이전트 테스트
│   ├── zero_agent.py               # 제로 에이전트 테스트
│   ├── list_envs.py                # 등록된 환경 목록
│   ├── convert_urdf.py             # URDF → USD 변환
│   └── rsl_rl/ sb3/ skrl/ rl_games/  # 학습 스크립트
└── source/rby1/rby1/
    ├── assets/                     # 로봇 URDF & 메쉬 (별도 다운로드)
    └── tasks/
        ├── direct/
        │   ├── rby1/               # Whole-Body Control
        │   ├── rby1_navigation/    # Navigation
        │   └── rby1_marl/          # Multi-Agent RL
        └── manager_based/
            └── rby1/               # Manager-Based Pose Control
```
