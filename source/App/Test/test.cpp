/***
* Program Name: adaBoost
*
* Script File: test.cpp
*
* Author: Joshua Zhang (zzbhf@mail.missouri.edu)
*
* Description:
*  
*  Testing phase
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
		"Usage:  test.exe test_file model_file predict_file \n"
		"press enter to continue...\n"
		);
	getchar();
	exit(-1);
}

int main(char argc, char* argv[])
{
	boostStruct		bst;

	if (argc < 4)
	{
		printf("Error: Not enough arguments!\n");
		exit_help();
	}

	boost_config(&bst);
	read_problem(argv[1], &bst);
	readModel(&bst, argv[2]);
	adaBoostTest(&bst, argv[3]);
	boost_destroy(&bst);

}

//int main()
//{
//	boostStruct		bst;
//
//	boost_config(&bst);
//	read_problem("C:/Users/Zhi/Desktop/test.dat", &bst);
//	readModel(&bst, "C:/Users/Zhi/Desktop/model.dat");
//	adaBoostTest(&bst, "C:/Users/Zhi/Desktop/predict.txt");
//	boost_destroy(&bst);
//
//}
