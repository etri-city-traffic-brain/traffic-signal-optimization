# SUMO를 활용한 강화학습 기반 신호 최적화

<table>
  <tr>
    <td><img src="assets/ft.gif?raw=true" width="100%"><p align="center">Fixed Time</p></td>
    <td><img src="assets/rl.gif?raw=true" width="100%"><p align="center">Reinforcement Learning</p></td>
  </tr>
</table>


## 실행 환경
- python 3.7
- keras 2.3.1
- tensorflow 2.0.0


## 사용법

- 모델 훈련
```
python run.py --mode train
```

- 모델 훈련 완료 후 테스트
```
python run.py --mode test --model-num xx
```

## References

1. _Traffic signal optimization for oversaturated urban networks: Queue growth equalization_, Jang et al., 2015
2. _Presslight: Learning max pressure control to coordinate traffic signals in arterial network_, Wei et al., 2019
