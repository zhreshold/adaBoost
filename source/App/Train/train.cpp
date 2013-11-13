/***
* Program Name: adaBoost
*
* Script File: train.cpp
*
* Author: Joshua Zhang (zzbhf@mail.missouri.edu)
*
* Description:
*  
*  Training phase
*   
*
* Copyright (C) 2013-2014.
* All Rights Reserved.
***/


#include <iostream>
#include <string.h>
#include "TLibCommon\readFile.h"


void exit_help()
{
	printf(
		"Usage: train.exe [options] train_file [model_file] \n"
		"options:\n"
		" -f fast thresholding mode: all features must be normalized to (0,1)\n"
		"   0 -- disable fast mode(default)\n"
		"   1 -- enable fast mode\n"
		" -s fast mode threholding pool size: quantized threshold number(default 100)\n"
		" -a accuracy: target accuracy(default 0.95)\n"
		" -i iteration: maximum iteration(default 1000)\n"
		"press enter to continue...\n"
		);
	getchar();
	exit(-1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name,  boostStruct* bst)
{
	int			i;
	Double		accuracy;
	ULong		maxIteration;
	int			mode, poolSize;
	char*		p;

	//parse options
	for ( i = 1; i < argc; i++)
	{
		if ( argv[i][0] != '-') break;
		if ( ++i >= argc)
			exit_help();
		switch(argv[i-1][1])
		{
		case 'a' : 
			accuracy = atof(argv[i]);
			bst->tarAccuracy = accuracy;
			break;
		case 'i' :
			maxIteration = atoi(argv[i]);
			bst->maxIter = maxIteration;
			break;
		case 'f' :
			mode = atoi(argv[i]);
			bst->fastMode = mode;
			break;
		case 's' :
			poolSize = atoi(argv[i]);
			bst->fastPoolSize = poolSize;
			break;
		default:
			fprintf(stderr, "unknown option: -%c\n", argv[i-1][1]);
			exit_help();
			break;
		}
	}

	//determine filenames

	if ( i >= argc)
		exit_help();
	strcpy(input_file_name, argv[i]);

	if ( i < argc-1)
		strcpy(model_file_name, argv[i+1]);
	else
	{
		p = strrchr(argv[i], '/');
		if ( p == NULL)
			p = argv[i];
		else
			p++;
		sprintf(model_file_name, "%s.model", p);
	}
}//end parse_command_line



int main(int argc, char* argv[])
{
	boostStruct		bst;
	char			input_file_name[1024], model_file_name[1024];

	boost_config(&bst);

	parse_command_line(argc, argv, &input_file_name[0], &model_file_name[0], &bst);

	read_problem(input_file_name, &bst);

	adaBoost(&bst);

	writeModel(&bst, model_file_name);

	boost_destroy(&bst);

}


//int main()
//{
//	boostStruct		bst;
//	
//
//	read_problem("C:/Users/Zhi/Desktop/test.dat", &bst);
//	
//	adaBoost(&bst);
//	writeModel(&bst, "../../image/model.dat");
//	boost_destroy(&bst);
//
//}