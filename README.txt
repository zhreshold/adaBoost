ADA-BOOST
-ADAPTIVE BOOSTING ALGORITHM

AUTHOR: JOSHUA ZHANG


TRAINING MODULE
----------------------------------------------------------------------------------

"Usage: train.exe [options] train_file [model_file] \n"
"options:\n"
" -f fast thresholding mode: all features must be normalized to (0,1)\n"
"   0 -- disable fast mode(default)\n"
"   1 -- enable fast mode\n"
" -s fast mode threholding pool size: quantized threshold number(default 100)\n"
" -a accuracy: target accuracy(default 0.95)\n"
" -i iteration: maximum iteration(default 1000)\n"



TESTING MODULE
----------------------------------------------------------------------------------

"Usage:  test.exe test_file model_file predict_file \n"

