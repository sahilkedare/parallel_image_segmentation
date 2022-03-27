#ifndef __UTIL_H__
#define __UTIL_H__
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <climits>
#include <float.h>
#define INF FLT_MAX
void getImg(char* name, float* img, int width, int height, int depth);
void getOutput(float* img, float* data, int width, int height, int depth, bool gpu, int s, int t, int r);
bool cmdOptionExists(char** begin, char** end, const std::string &option);
char* getCmdOption(char** begin, char** end, const std::string &option);
void serialquickshift(float* img, float* enrg, int width, int height, int depth,
	int sigma3, int tau2, int ratio, float* parent, float* dist);
#endif
