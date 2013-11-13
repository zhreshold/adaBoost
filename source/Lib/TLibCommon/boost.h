/***
* Program Name: adaBoost
*
* Script File: boost.h
*
* Author: Joshua Zhang (zzbhf@mail.missouri.edu)
*
* Description:
*  
*  Boost class header
*   
*
* Copyright (C) 2013-2014.
* All Rights Reserved.
***/

#ifndef __BOOST_H__
#define __BOOST_H__

#include "define.h"
#define	 DEFAULT_FEAT_LENGTH		10000
#define	 DEFAULT_MAX_ITERATION		1000
#define	 DEFAULT_STOP_CRITERIA		0.001
#define	 DEFAULT_TARGET_ACCURACY	0.95

//fast mode -- requirement: features are normalized to (0,1)
#define	 DEFAULT_USE_FAST_MODE		0
#define	 DEFAULT_FAST_POOL_SIZE		100


typedef struct boostStructure
{
	ULong			numPos, numNeg;
	ULong			numTotal;
	UInt			iter;
	Double			minErr;

	Bool			fastMode;
	UInt			fastPoolSize;
	

	Double			tarAccuracy;
	ULong			featLength;
	UInt			maxIter;
	Double			stopCriteria;

	ULong*			posSelection;
	Double*			thresholds;
	char*			signs;
	Double*			accuracy;
	Double*			beta;
	Double*			alpha;

	Double**		weights;
	FeatureType**	features;
	Bool*			labels;

	ULong*			poolSize;
	FeatureType**	thresholdPool;
	FeatureType*	thresholdPoolTmp;

}boostStruct;


void boost_config(boostStruct* bst, ULong featLength = DEFAULT_FEAT_LENGTH, ULong maxIter = DEFAULT_MAX_ITERATION, 
			Double tarAccuracy = DEFAULT_TARGET_ACCURACY, Double stopCriteria = DEFAULT_STOP_CRITERIA);

void boost_init(ULong numPos, ULong numNeg, boostStruct* bst);
void boost_destroy(boostStruct* bst);
void adaBoost(boostStruct* bst);
void writeModel(boostStruct* bst, const char* modelFile);
void readModel(boostStruct* bst, const char* modelFile);
void adaBoostTest(boostStruct* bst, const char* predictOutput);


#endif // _BOOST_H_
