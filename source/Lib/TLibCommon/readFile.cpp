/***
* Program Name: adaBoost
*
* Script File: readFile.cpp
*
* Author: Joshua Zhang (zzbhf@mail.missouri.edu)
*
* Description:
*  
*  Read file
* 
*
* Copyright (C) 2013-2014.
* All Rights Reserved.
***/

#include "readFile.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>


static char *line = NULL;
static int max_line_len;

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


// read in a problem (in libsvm format)
void read_problem(const char *filename, boostStruct* bst)
{
	ULong		numInstance, featLength, temp, numPos, numNeg;
	FILE*		fp;
	char*		endptr;
	char*		idx, *val, *label;
	char*		p;
	int			label1, label2, tmpLabel;
	int			i;

	fp = fopen(filename, "r");
	if ( fp == NULL)
	{
		fprintf(stderr,"Can't open input file %s\n", filename);
		exit(-1);
	}

	numInstance = 0;
	featLength = 0;
	max_line_len = 1024;
	label1 = 1;
	label2 = 0;
	line = (char*)malloc(sizeof(char) * max_line_len);

	while( readline(fp) != NULL)
	{
		p = strtok(line, " \t"); //label
		tmpLabel = (int)strtol(p, &endptr, 10);
		if ( numInstance == 0)
		{
			label1 = tmpLabel;
			label2 = tmpLabel;
		}
		else if ( label1 != tmpLabel)
			label2 = tmpLabel;

		//features
		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");
			if ( val == NULL || *val == '\n')
				break;
			if ( strcmp(idx, "qid") != 0)
			{
				temp = (ULong) strtol(idx, &endptr, 10);
				if ( temp > featLength)
					featLength = temp;
			}

		}
		numInstance++;
	}
	rewind(fp);

	if (label1 == label2)
	{
		fprintf(stderr, "Only 1 class found!\n");
		exit(-1);
	}
	else if ( label1 < label2)
	{
		tmpLabel = label1;
		label1 = label2;
		label2 = tmpLabel;
	}

	numPos = numNeg = 0;

	while(readline(fp) != NULL)
	{
		p = strtok(line, "\t\n");
		tmpLabel = (int)strtol(p, &endptr, 10);
		if ( tmpLabel == label1)
			numPos++;
		else if (tmpLabel == label2)
			numNeg++;
	}
	rewind(fp);

	//boost_config(bst, featLength);
	bst->featLength = featLength;
	boost_init(numPos, numNeg, bst);

	for ( i = 0; i < numInstance; i++)
	{
		readline(fp);
		label = strtok(line, " \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		tmpLabel = strtod(label, &endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);
		if ( tmpLabel == label1)
			bst->labels[i] = 1;
		else
			bst->labels[i] = 0;

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			if (strcmp(idx,"qid") == 0)
			{
				continue;
			}
			errno = 0;
			temp = (ULong)strtol(idx, &endptr, 10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || temp > featLength)
				exit_input_error(i+1);
			
			errno = 0;
			bst->features[i][temp-1] = (FeatureType)strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);
		}
	}

	free(line);
	fclose(fp);
}



