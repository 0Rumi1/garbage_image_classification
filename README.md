# [이미지 분류_딥러닝] 쓰레기 분리수거를 위한 이미지 분류

## 프로젝트 소개
쓰레기 분리수거 시, 쓰레기의 종류에 따라 어떻게 분류해야할 지 모르는 경우가 종종 발생한다.
이러한 불편함을 해소하기 위해 **딥러닝을 활용해 쓰레기 분리수거 모델 개발** 하는 프로젝트를 기획하게 됨
  <br>

* 멤버: 본인(1명)
* 역할: 전체 기획 및 이미지 딥러닝 모델 구축
  <br>
* 개발 기간: 23.03.26-27
  <br>

**개발 방향 및 목표는 아래와 같다.**
1. 이미지 분류로 쓰레기 분리수거 딥러닝 모델 구축 및 개발
2. 분류 모델 성능: 90% 이상
3. 쓰레기 분리수거 이미지 업로드 시, 분리수거 라벨 메시지 출력 -> 시간상 관계로 개발하지 못함
4. 추후, 휴대폰 카메라로 쓰레기를 촬영하여 이미지 판별 후 분리수거 카테고리를 안내하는 메세지를 출력하는 서비스 개발
  <br>

## 작업 순서
* 데이터 불러오기
* 이미지 전처리
* 베이스라인 모델 설정
* 모델 튜닝 후 성능 높이기
  * 데이터 증강
  * 전이 학습
* 모델 성능 확인
* 결론
  <br>

## 데이터 구조
* 라벨: 12개
* 실제 사용 라벨은 11개 (plastic 제외) 제외 이유는 학습 시, **'plastic 라벨의 이미지 파일이 존재하지 않는다'** 오류 발생으로 해당 라벨을 삭제 후 학습하는 것으로 결정함


garbage_classification

 └ battery

 └ biological

 └ brown-glass

 └ cardboard

 └ clothes

 └ green-glass

 └ metal

 └ paper
 
 └ plastic
 
 └ shoes
 
 └ trash
 
 └ white-glass
  <br>

## 기술 스택
#### Environment
<img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"/>
<img src="https://img.shields.io/badge/windows-0078D6?style=for-the-badge&logo=windows&logoColor=white"/>


#### Development
 <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 
  <br>

---

## 목차
1. [구현 기능](#구현-기능)
2. [사용법](#사용법)
3. [결과](#결과)
4. [배운점 & 아쉬운 점](#배운점-&-아쉬운-점)
5. [이후의 계획](#이후의-계획)
  <br>

## 구현 기능
1. 딥러닝 기반의 쓰레기 이미지 분류 모델 구축
<br>

## 사용법
**zip 이미지 파일 압축 해제 방법**
압축 파일의 위치(구글 드라이브 - 내 드라이브 위치)
```
drive_path = '/content/drive/MyDrive/Colab Notebooks/project/'
source_filename = drive_path + 'dataset/archive (5).zip'
```
  <br>

저장할 경로
```
extract_folder = '/content/drive/MyDrive/Colab Notebooks/project/dataset/'
```
  <br>

압축 해제
```
import shutil
shutil.unpack_archive(source_filename, extract_folder)
```
  <br>

```
import os
import matplotlib.pyplot as plt


def plot_imags(item_dir, top = 10):
  all_trash_dirs = os.listdir(item_dir)
  item_files = [os.path.join(item_dir, file) for file in all_trash_dirs][:5]

  plt.figure(figsize = (10,10))

  for idx, image_path in enumerate(item_files):
    sp = plt.subplot(5,5, idx+1)

    img = plt.imread(image_path)
    plt.tight_layout()
    sp.axis('Off')
    plt.imshow(img, cmap = 'gray')
```

```
plot_imags(data_path + '/battery')
plot_imags(data_path + '/biological')
plot_imags(data_path + '/brown-glass')
```



* 테스트 방법
  <br>

## 결과
3-1) 모델 정확도 67.9% 로 아쉬운 결과를 나타냄

![image](https://user-images.githubusercontent.com/122415320/235335209-b12f9abe-8fc1-45cb-8ba2-e818aefc01c5.png)

3-2) Train/Test 모델 손실 및 정확도 그래프
![image](https://user-images.githubusercontent.com/122415320/235335200-0b291aec-0bc4-418b-acf3-0d2668fd2c7a.png)
  <br>


## 배운점 & 아쉬운 점
  <br>
  
 * 데이터셋 파일을 구글 드라이브에 저장하지 않은 이유는 15,000장의 파일을 구글 드라이브를 통해 읽어오면 저장하는 속도가 매우 느리기 때문이라는 사실을 알게됨
  
 * 따라서, 구글 드라이브의 압축 파일 한 개를 읽어오는 시간은 오래 걸리지 않고, 압축을 풀면서 생성되는 15,000장의 파일을 코랩 환경에 저장하면 파일을 읽는 시간이 훨씬 단축됨

## 이후의 계획
  <br>


