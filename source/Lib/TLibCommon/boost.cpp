/***
* Program Name: adaBoost
*
* Script File: boost.cpp
*
* Author: Joshua Zhang (zzbhf@mail.missouri.edu)
*
* Description:
*  
*  Boosting 
*   
*
* Copyright (C) 2013-2014.
* All Rights Reserved.
***/

#include "boost.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>


// Configuration
void boost_config(boostStruct* bst, ULong featLength, ULong maxIter, Double tarAccuracy, Double stopCriteria)
{
	bst->featLength = featLength;
	bst->maxIter = maxIter;
	bst->stopCriteria = stopCriteria;
	bst->tarAccuracy = tarAccuracy;
	bst->fastMode = DEFAULT_USE_FAST_MODE;
	bst->fastPoolSize = DEFAULT_FAST_POOL_SIZE;
}
//end config


// initilization
void boost_init(ULong numPos, ULong numNeg, boostStruct* bst)
{
	ULong	 i;

	bst->numPos = numPos;
	bst->numNeg = numNeg;
	bst->numTotal = numPos + numNeg;


	bst->iter = 0;
	
	//memory allocation
	bst->posSelection = (ULong*)malloc(sizeof(ULong) * (bst->maxIter+1));
	bst->thresholds = (Double*)malloc(sizeof(Double) * (bst->maxIter+1));
	bst->signs = (char*)malloc(sizeof(char) * (bst->maxIter+1));
	bst->accuracy = (Double*)malloc(sizeof(Double) * (bst->maxIter+1));
	bst->beta = (Double*)malloc(sizeof(Double) * (bst->maxIter+1));
	bst->alpha = (Double*)malloc(sizeof(Double) * (bst->maxIter+1));

	bst->weights = (Double**)malloc(sizeof(Double*) * (bst->maxIter+1));
	for ( i = 0; i < (bst->maxIter+1); i++)
	{
		bst->weights[i] = (Double*)malloc(sizeof(Double) * bst->numTotal);
	}
	bst->features = (FeatureType**)malloc(sizeof(FeatureType*) * bst->numTotal);
	for ( i = 0; i < bst->numTotal; i++)
	{
		bst->features[i] = (FeatureType*)malloc(sizeof(FeatureType) * bst->featLength);
		memset(&bst->features[i][0], 0, sizeof(FeatureType) * bst->featLength);
	}
	bst->labels = (Bool*)malloc(sizeof(Bool)*bst->numTotal);

	//threshold pool
	bst->poolSize = (ULong*)malloc(sizeof(ULong) * bst->featLength);
	bst->thresholdPool = (FeatureType**)malloc(sizeof(FeatureType*) * bst->numTotal);
	for ( i = 0; i < bst->numTotal; i++)
	{
		bst->thresholdPool[i] = (FeatureType*)malloc(sizeof(FeatureType) * bst->featLength);
		memset(&bst->thresholdPool[i][0], 0, sizeof(FeatureType) * bst->featLength);
	}
	bst->thresholdPoolTmp = (FeatureType*)malloc(sizeof(FeatureType) * bst->numTotal);
	memset(bst->thresholdPoolTmp, 0, sizeof(FeatureType) * bst->numTotal);


}//end init


void boost_destroy(boostStruct* bst)
{
	ULong	 i;

	free(bst->posSelection);
	free(bst->thresholds);
	free(bst->signs);
	free(bst->accuracy);
	free(bst->beta);
	free(bst->alpha);

	free(bst->poolSize);
	for ( i = 0; i < bst->numTotal; i++)
	{
		free(bst->thresholdPool[i]);
	}
	free(bst->thresholdPool);
	free(bst->thresholdPoolTmp);

	for ( i = 0; i < (bst->maxIter+1); i++)
	{
		free(bst->weights[i]);
	}
	free(bst->weights);

	for ( i = 0; i < bst->numTotal; i++)
	{
		free(bst->features[i]);
	}
	free(bst->features);
	free(bst->labels);



}//end destroy

// Initilize weights with 1/2m and 1/2l
void initWeights(boostStruct* bst)
{
	ULong		i;

	for ( i = 0; i < bst->numTotal; i++)
	{
		if ( bst->labels[i])
		{
			bst->weights[0][i] = 0.5 / bst->numPos;
		}
		else
		{
			bst->weights[0][i] = 0.5 / bst->numNeg;
		}
	}


}//end initWeights

// Normalize Weights before each iteration
void normWeights(boostStruct* bst)
{
	ULong	i;
	Double	sum;

	sum = 0;
	for ( i = 0; i < bst->numTotal; i++)
	{
		sum += bst->weights[bst->iter][i];
	}

	for ( i = 0; i < bst->numTotal; i++)
	{
		bst->weights[bst->iter][i] /= sum;
	}

}//end normWeigths






inline int weakOutput(FeatureType x, Double thresh, char sign,  Bool label)
{
	if ( sign == 1) 
		return ( (x < thresh) ^ label);
	else
		return ( (x > thresh) ^ label);
}


void calcError(boostStruct* bst, Double* err, ULong j, Double thresh, char* sign)
{
	ULong	i;

	Double	error;

	error = 0;

	for ( i = 0; i < bst->numTotal; i++)
	{
		error += bst->weights[bst->iter][i] * weakOutput(bst->features[i][j], thresh, 1, bst->labels[i]);
	}

	if (error < 0.5)
	{
		*err = error;
		*sign = 1;
	}
	else
	{
		*err = 1-error;
		*sign = -1;
	}
	//error = 0;

	//for ( i = 0; i < bst->numTotal; i++)
	//{
	//	error += bst->weights[bst->iter][i] * weakOutput(bst->features[i][j], thresh, 0, bst->labels[i]);
	//}

	//if (error < *err)
	//{
	//	*err = error;
	//	*sign = -1;
	//}


}//end calcError

// inline comparison function for qsort
inline int myCmpFunc(const void* a, const void* b)
{
	return (*(FeatureType *)a > *(FeatureType *)b)? 1 : -1;
}


void thresholdPool(boostStruct* bst)
{
	ULong		i, j, poolSize;

	for ( j= 0; j < bst->featLength; j++)
	{
		for ( i = 0; i < bst->numTotal; i++)
		{
			bst->thresholdPoolTmp[i] = bst->features[i][j];
		}
		qsort(bst->thresholdPoolTmp, bst->numTotal, sizeof(FeatureType), myCmpFunc);

		poolSize = 0;
		bst->thresholdPool[0][j] = bst->thresholdPoolTmp[0];

		for ( i = 0; i < bst->numTotal; i++)
		{
			if ( bst->thresholdPool[poolSize][j] < bst->thresholdPoolTmp[i])
			{
				poolSize++;
				bst->thresholdPool[poolSize][j] = bst->thresholdPoolTmp[i];
			}
		}
		poolSize++;	

		bst->poolSize[j] = poolSize;
	}
}//end thresholdPool

void fastThresholdPool(boostStruct *bst)
{
	ULong		i, j;
	ULong		poolSize;

	poolSize = bst->fastPoolSize;

	if ( poolSize > bst->numTotal)
	{
		thresholdPool(bst);
		return;
	}


	for ( i = 0; i < bst->featLength; i++)
	{
		bst->poolSize[i] = poolSize;
		for ( j = 0; j < poolSize; j++)
		{
			bst->thresholdPool[j][i] = (double) j / poolSize;
		}
	}
}

	


// find the best weak classifier
void selectWeakClassifier(boostStruct* bst)
{
	ULong			i, j, poolSize;
	Double			minErr, tmpThresh, tmpErr;
	char			tmpSign;
	FeatureType		minVal, maxVal;

	minErr = DBL_MAX;
	
	for ( j = 0; j < bst->featLength; j++)
	{
		//find min value and max value
		//minVal = bst->features[0][j];
		//maxVal = bst->features[0][j];
		//for ( i = 1; i < bst->numTotal; i++)
		//{
		//	if ( bst->features[i][j] < minVal)
		//		minVal = bst->features[i][j];
		//	if ( bst->features[i][j] > maxVal)
		//		maxVal = bst->features[i][j];
		//}

		poolSize = bst->poolSize[j];

		//update errors by adjusting threshold in every step
		for ( i = 0; i < poolSize; i++)
		//for ( tmpThresh = minVal; tmpThresh < maxVal; tmpThresh += THRESHOLD_STEP_SIZE)
		{
			tmpThresh = bst->thresholdPool[i][j];
			calcError(bst, &tmpErr, j, tmpThresh, &tmpSign);
			if (tmpErr < minErr)
			{
				minErr = tmpErr;
				bst->thresholds[bst->iter] = tmpThresh;
				bst->signs[bst->iter] = tmpSign;
				bst->posSelection[bst->iter] = j;
				bst->minErr = tmpErr;
			}
		}
	}


}

void updateWeights(boostStruct* bst)
{
	ULong	i;
	Double  beta;
	ULong	pos;
	Double	threshold;
	char	sign;
	Double	err;

	pos = bst->posSelection[bst->iter];
	threshold = bst->thresholds[bst->iter];
	sign = bst->signs[bst->iter];
	err = bst->minErr;

	beta = err / ( 1 - err);
	bst->beta[bst->iter] = beta;
	bst->alpha[bst->iter] = log(1 / beta);

	for ( i = 0; i < bst->numTotal; i++)
	{
		if ( weakOutput( bst->features[i][pos], threshold, sign, bst->labels[i]))
		{
			//classified incorrectly
			bst->weights[bst->iter+1][i] = bst->weights[bst->iter][i];
		}
		else
		{
			//classified correctly
			bst->weights[bst->iter+1][i] = bst->weights[bst->iter][i] * beta;
		}
	}
}//end updateWeights


inline int weakClassifier(FeatureType x, Double thresh, char sign)
{
	if ( sign == 1)
		return ( x < thresh);
	else
		return ( x > thresh);
}

int	strongClassifier(boostStruct* bst, ULong idx)
{
	int			i;
	Double		left, right;

	left = 0;
	right = 0;

	for ( i = 0; i <= bst->iter; i++)
	{
		left += bst->alpha[i] * weakClassifier( bst->features[idx][bst->posSelection[i]], bst->thresholds[i], bst->signs[i]);
		right += bst->alpha[i];
	}

	right *= 0.5;

	if ( left >= right)
		return 1;
	else
		return 0;

}//end strongClassifier


void computeAccuracy(boostStruct* bst)
{
	ULong	i;
	ULong	numCorrect, numIncorrect;
	Double	accuracy;
	int		tmpLabel;

	numCorrect = numIncorrect = 0;
	
	for ( i = 0; i < bst->numTotal; i++)
	{
		tmpLabel = strongClassifier(bst, i);
		if ( tmpLabel == bst->labels[i])
			numCorrect++;
		else
			numIncorrect++;
	}
	
	accuracy = (double)numCorrect / (numCorrect + numIncorrect);
	bst->accuracy[bst->iter] = accuracy;

}//end computeAccuracy



void adaBoost(boostStruct* bst)
{
	//init 
	initWeights(bst);
	bst->iter = 0;

	//generate threshold pool
	//thresholdPool(bst);
	if ( bst->fastMode )
	{
		fastThresholdPool(bst);
	}
	else
	{
		thresholdPool(bst);
	}

	do
	{
		printf("Iter: %d ", bst->iter);
		normWeights(bst);
		selectWeakClassifier(bst);
		printf("Threshold: %f Position: %d Sign: %d\n", bst->thresholds[bst->iter], bst->posSelection[bst->iter], bst->signs[bst->iter]);
		updateWeights(bst);
		computeAccuracy(bst);
		printf("Accuracy: %f\n", bst->accuracy[bst->iter]);
		bst->iter++;
	}while( (bst->iter < (bst->maxIter)) && (bst->accuracy[bst->iter-1] < bst->tarAccuracy));



}


void writeModel(boostStruct* bst, const char* modelFile)
{
	FILE*		fp;
	ULong		i, j;


	fp = fopen(modelFile, "wt");
	if ( fp == NULL)
	{
		printf("Error open model file to write!\n");
		exit(-1);
	}

	fprintf(fp, "%d\n", bst->iter);
	
	for ( i = 0; i < bst->iter; i++)
	{
		fprintf(fp, "%d,%lf,%d,%lf\n", bst->posSelection[i], bst->thresholds[i], bst->signs[i], bst->alpha[i]);
	}

	fclose(fp);

}//end writeModel


void readModel(boostStruct* bst, const char* modelFile)
{
	FILE*		fp;
	ULong		iter, pos, i;
	char		sign;
	Double		threshold, alpha;

	fp = fopen(modelFile, "rt");
	if ( fp == NULL)
	{
		printf("Error open model file to read!\n");
		exit(-1);
	}

	fscanf(fp, "%d", &iter);

	bst->iter = iter;
	

	if ( iter > bst->maxIter)
	{
		//re-allocate memory
		bst->posSelection = (ULong*)realloc(bst->posSelection, sizeof(ULong) * (bst->iter+1));
		bst->thresholds = (Double*)realloc(bst->thresholds, sizeof(Double) * (bst->iter+1));
		bst->signs = (char*)realloc(bst->signs, sizeof(char) * (bst->iter+1));
		bst->accuracy = (Double*)realloc(bst->accuracy, sizeof(Double) * (bst->iter+1));
		bst->beta = (Double*)realloc(bst->beta, sizeof(Double) * (bst->iter+1));
		bst->alpha = (Double*)realloc(bst->alpha, sizeof(Double) * (bst->iter+1));

		for ( i = 0; i < (bst->maxIter+1); i++)
		{
			free(bst->weights[i]);
		}
		bst->weights = (Double**)realloc(bst->weights, sizeof(Double*) * (bst->iter+1));
		for ( i = 0; i < (bst->iter+1); i++)
		{
			bst->weights[i] = (Double*)malloc(sizeof(Double) * bst->numTotal);
		}
		bst->maxIter = iter;
	}
	
	bst->iter--;
	//load parameters
	for ( i = 0; i <= iter; i++)
	{
		fscanf(fp, "%d,%lf,%d,%lf\n", &bst->posSelection[i], &bst->thresholds[i], &bst->signs[i], &bst->alpha[i]);
	}

	fclose(fp);

}//end readModel



void adaBoostTest(boostStruct* bst, const char* predictOutput)
{
	ULong	i;
	ULong	numCorrect, numIncorrect;
	Double	accuracy;
	int		tmpLabel;
	FILE*	fp;


	fp = fopen(predictOutput, "wt");

	if ( fp == NULL)
	{
		printf("Error open predict file!\n");
		exit(-1);
	}

	numCorrect = numIncorrect = 0;
	
	for ( i = 0; i < bst->numTotal; i++)
	{
		tmpLabel = strongClassifier(bst, i);
		fprintf(fp, "%d\n", tmpLabel);

		if ( tmpLabel == bst->labels[i])
			numCorrect++;
		
		else
			numIncorrect++;
	}
	
	accuracy = (double)numCorrect / (numCorrect + numIncorrect);

	printf("Accuracy: %f\n", accuracy);

	fclose(fp);

}//end adaBoostTest