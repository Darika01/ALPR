# Haar/LBP cascade training with OpenCV

### Train your own cascade
Prepare positive and negative samples according to the [instruction](https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html) (you can use *cascade_train_data/generate_data.py* to generate Good.dat and Bad.dat from your samples) or use existing files from *cascade_train_data/* folder and run the *opencv_traincascade.exe* program with the relevant data from the console:

```bash
opencv_traincascade.exe -data haarcascade\ -vec cascade\samples.vec -bg Bad.dat -numStages 16
-minHitRate 0.995 -maxFalseAlarmRate 0.4 -numPos 260 -numNeg 598-w 120 -h 30 -mode ALL -preca
lcValBufSize 1024 precalcIdxBufSize 1024
```