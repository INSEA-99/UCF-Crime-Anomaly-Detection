# UCF-Crime-Anomaly-Detection
https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
#### 개요
-  CCTV의 설치는 날이 갈수록 증가하고 있다. 하지만 사고 예방 측면보단 사건 발생 후 증거자료로서의 활용 측면이 강하다. 수많은 CCTV를 24시간 감시할 인력을 확보하는 것은 현실적으로 불가능하기 때문이다. 따라서 위험한 상황을 Anomaly Detection model을 통해 자동적으로 인지하여 경고해준다면 사고 예방 측면 활용도가 높아질 것이다. CCTV는 모든 산업분야에서 사용하고 있기 때문에, 현실화된다면 큰 효용성을 보일 것이다.

# Getting Started

## Dataset
UCF-Crime Video Dataset
> Nomal Video와 13가지의 카테고리로 라벨링된 Anomaly Video로 구성되어 있다.
https://www.crcv.ucf.edu/projects/real-world/

## Anomaly Detection Model

![image](https://user-images.githubusercontent.com/61490878/173889113-bdc726d4-f2a0-4c7c-a366-a73d041a4e38.png)

1) weakly labeling 된 video data를 사용하고 3d Convolution(C3D, I3D, R3D)를 통해 segments 별 feature을 추출한다. 
2) 추출된 feature을 Dense Layers를 통해 anomaly score를 계산한다.
3) MIL Ranking Loss 함수와 smoothness constraints로 이루어진 목적함수를 이용해 Anomaly score를 최적화한다.
4) MIL Ranking Loss 함수를 top-k mean MIL Ranking Loss로 수정하여
      성능을 비교한다.


## result
#### 1) 3D convolution 구조 별 성능 비교(기존 목적함수 사용)
![image](https://user-images.githubusercontent.com/61490878/173891325-1783e95b-6cd4-4def-80e0-271e141a1795.png)

 
 
#### 2) 수정된 목적함수 사용했을 때 성능 비교(I3D, R3D)
![image](https://user-images.githubusercontent.com/61490878/173891487-48f19a1e-7de5-451f-8131-477d0a774e65.png)
> I3D, R3D 구조의 목적함수 별 성능 비교(top-k의 평균을 사용했으며, W로 표시된 것은 가중평균을 사용한 것이다. 가중치는 높은 순으로 각각 top-3은  0.5, 0.3, 0.2, top-4는 0.4, 0.3, 0.2, 0.1, top-5는 0.3, 0.3, 0.2, 0.1, 0.1, top-6은 0.3, 0.3, 0.1, 0.1, 0.1, 0.1을 사용하였다.)


#### 3) 기존 연구와 성능 비교

![image](https://user-images.githubusercontent.com/61490878/173891100-ac3dc036-ec15-4c10-a505-0f88616121a2.png)




## conclusion
 3d-convolution 구조와 목적함수의 수정을 통해 성능을 향상시킬 수 있었다. 목적함수는 anomaly score가 가장 큰 값만 반영하는 기존 MIL ranking function과 anomaly score가 가장 큰 k개의 값 평균을 사용하는 top-k mean function을 사용하였는데, 두 가지 모두 동영상의 짧고 긴 시간적 의존성을 잘 담아내지 못한다는 한계점이 있다. 이로 인해 짧은 anomaly event를 포함하는 경우 검출하지 못한다는 문제점을 나타낸다. 따라서 시간적 정보를 보완해주는 구조의 추가가 필요해 보인다.

## 참고문헌
가. Sultani, W., Chen, C., & Shah, M. (2018). Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 6479-6488).

나. Tian, Y., Pang, G., Chen, Y., Singh, R., Verjans, J. W., & Carneiro, G. (2021). Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4975-4986).
