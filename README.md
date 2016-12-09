#ADA-BOOST
Adaptive boosting algorithm implementation in C++


## Usage 
*TRAINING MODULE*
```
Usage: train.exe [options] train_file [model_file]
options:
 -f fast thresholding mode: all features must be normalized to (0,1)
   0 -- disable fast mode(default)
   1 -- enable fast mode
 -s fast mode threholding pool size: quantized threshold number(default 100)
 -a accuracy: target accuracy(default 0.95)
 -i iteration: maximum iteration(default 1000)
```


*TESTING MODULE*
```
Usage:  test.exe test_file model_file predict_file
```
