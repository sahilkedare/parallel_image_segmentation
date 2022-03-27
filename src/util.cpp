#include "util.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define MIN_PIXELS 20

void getImg(char* name, float* img, int width, int height, int depth)
{
	// generate image.png
	unsigned char* image_char = (unsigned char*) malloc(width * height *
		depth * sizeof(unsigned char));
	float * offset_float;
	for(int i = 0; i < height; ++i){
		for(int j = 0; j < width; ++j){
			offset_float = img + i * width + j;
			for(int k = 0; k < depth; ++k){
				// one pixel contains 3 8-bit units, each unit is an ascii char.
				image_char[i * width * depth + j * depth + k] =
				(char) *(offset_float + k * width * height); // numerate?
			}
		}
	}
	stbi_write_png(name, width, height, depth, image_char,
		width * depth * sizeof(unsigned char));

}

void getOutput(float* img, float* data, int width, int height, int depth, bool gpu, int s, int t, int r)
{
	// get the flat graph
	int size = width * height;
	int *map = (int*) malloc(size * sizeof(int));
	int num = 0;
	for(int i = 0; i < size; ++i)
	{
		map[i] = data[i];
	}

	bool flag = true;
	while(flag)
	{
		flag = false;
		for(int i = 0; i < size; ++i)
		{
			flag = flag || (map[i] != map[map[i]]);
			map[i] = map[map[i]];
		}
	}

	// get mean RGB values of each tree
	float* mean = (float*) calloc(size * depth, sizeof(float));
	float* count = (float*) calloc(size, sizeof(float));
	for(int i = 0; i < size; ++i)
	{
		count[map[i]]++;
		for(int k = 0; k < depth; ++k)
		{
			mean[map[i] + k * width * height] += img[i + k * width * height];
		}
	}

	/*
	// group segment smaller than MIN_PIXELS
	for(int i = 0; i < size; ++i)
	{
		if(count[i] < MIN_PIXELS)
		{
			int bias = 1;
			while(i + bias < size && count[i + bias] < MIN_PIXELS &&
					i > bias && count[i - bias] < MIN_PIXELS)
			{
				bias++;
			}
			if(count[i + bias] >= MIN_PIXELS)
			{
				map[i] = map[i + bias];
			}
			else
			{
				map[i] = map[i - bias];
			}
		}
	}*/

	for(int i = 0 ; i < size; ++i)
	{
		for(int k = 0; k < depth; ++k)
		{
			mean[i + k * width * height] /= count[i];
		}
	}
	free(count);
	
	// colorfy segments
	for(int i = 0; i < size; ++i)
	{
		for(int k = 0; k < depth; ++k)
		{
			mean[i + k * width * height] = mean[map[i] + k * width * height];
		}
	}

	// display superpixel boundary
	for(int i = 0; i < size; ++i)
	{
		if(map[i] == i)
		{
			num++;/*
			for(int k = 0; k < depth; ++k)
			{
				mean[i + k * width * height] = 0.0;
			}*/
		}
	}
	printf("num of superpixel is %d.\n", num);

	getImg("test_gpu.png", mean, width, height, depth);
	free(count);
	free(map);
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

char* getCmdOption(char** begin, char** end, const std::string& option)
{
	char** it = std::find(begin, end, option);
	if(it != end && ++it != end)
	{
		return *it;
	}
	return 0;
}

void serialquickshift(float* img, float* enrg,
	int width, int height, int depth,
	int sigma3, int tau2, int ratio,
	float* parent, float* dist)
{
	int x_start;
	int y_start;
	int x_end;
	int y_end;
	float temp;
	float atom;
	float E;
	// get enrg for each pixel
	for(int i = 0; i < height; ++i)
	{
		for(int j = 0; j < width; ++j)
		{
			E = 0;
			x_start = i < sigma3 ? 0 : i - sigma3;
			x_end = i + sigma3 > height ? height : i + sigma3;
			y_start = j < sigma3 ? 0 : j - sigma3;
			y_end = j + sigma3 > width ? width : j + sigma3;
			for(int m = x_start; m < x_end; ++m)
			{
				for(int n = y_start; n < y_end; ++n)
				{
					temp = 0;
					// get distance
					for(int k = 0; k < depth; ++k)
					{
						atom =  img[m * width + n + k * width * height] -
										img[i * width + j + k * width * height];
						temp += atom * atom;
					}
					atom = ratio * (m - i);
					temp += atom * atom;
					atom = ratio * (n - j);
					temp += atom * atom;
					E += exp(-temp * 4.5 / sigma3 / sigma3);
				}
			}
			enrg[i * width + j] = E / (x_end - x_start) / (y_end - y_start);
		}
	}

	int x;
	int y;
	// get dist for each pixel
	for(int i = 0; i < height; ++i)
	{
		for(int j = 0; j < width; ++j)
		{
			E = enrg[i * width + j];
			dist[i * width + j] = INF;
			x = i;
			y = j;
			x_start = i < sigma3 ? 0 : i - sigma3;
			x_end = i + sigma3 > height ? height : i + sigma3;
			y_start = j < sigma3 ? 0 : j - sigma3;
			y_end = j + sigma3 > width ? width : j + sigma3;
			for(int m = x_start; m < x_end; ++m)
			{
				for(int n = y_start; n < y_end; ++n)
				{
					if(enrg[m * width + n] > E)
					{
						temp = 0;
						for(int k = 0; k < depth; ++k)
						{
							atom = img[m * width + n + k * width * height] -
											img[i * width + j + k * width * height];
							temp += atom * atom;
						}
						atom = ratio * (m - i);
						temp += atom * atom;
						atom = ratio * (n - j);
						temp += atom * atom;
						if(temp < dist[i * width + j] && temp < tau2)
						{
							dist[i * width + j] = temp;
							x = m;
							y = n;
						}
					}
				}
			}
			parent[i * width + j] = x * width + y;
			dist[i * width + j] = sqrt(dist[i * width + j]);
		}
	}
	return;
}
