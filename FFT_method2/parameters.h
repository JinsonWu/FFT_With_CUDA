#pragma once
#define SIZE 	2048
#define LOOP	4

class IndexSave
{
public:
	int blockInd_x;
	int threadInd_x;
	int head;
	int stripe;
};