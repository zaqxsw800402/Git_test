# README

# YT_gesture_control

### 簡介

這份專題是在AI課程與其他同學們一起製作的，藉由攝影機辨別手的姿勢來操控電腦鍵盤及滑鼠

### 流程圖

![README%2072dc6f27ee414200ba37ad633a3f41ad/Untitled.png](README%2072dc6f27ee414200ba37ad633a3f41ad/Untitled.png)

### 模型訓練

經由openCV來抓取攝影機的畫面，並藉由mediapipe來偵測手的骨架及標出相對位置，運用這些相對位置來訓練出能辨別手勢模型

### 操控鍵盤及滑鼠

辨別出來的手勢(類別)可以藉由pyautogui來執行操控

### Monitor

設置成永遠置頂，並會隨著模式的不同改變顏色，並在30秒內沒偵測到手會轉回到休息模式

### 操作說明

能辨別的手勢及相對應的電腦操作已經放在introduction裡，

或是藉由觀看[youtube](https://www.youtube.com/watch?v=3t_t8A_DUxU)來學習

# Little Project

這是我在線上平台[datacamp](https://learn.datacamp.com/)裡做的小專題