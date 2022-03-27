// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"
#include "quickshift.h"
#include "util.h"
#include <sys/time.h>
#include <string>

int main(int argc, char **argv)
{
	if(cmdOptionExists(argv, argv + argc, "-h") ||
		 cmdOptionExists(argv, argv + argc, "--h"))
	{
		printf("This program implements quickshift for image segmentation.");
		printf("Only support for PNG image.\ndefault value:\nsigma: 6\ntau: 10\n");
		printf("ration: 1\nYou may change parameters with '--sigma', '--tau',");
		printf("'--ratio'");
	}

	// set default values
	float sigma = 6;
	float tau = 10;
	float ratio = 1;
	char* input = "test.png";

	// get cmd values
	if(cmdOptionExists(argv, argv + argc, "--sigma"))
	{
		char* temp = getCmdOption(argv, argv + argc, "--sigma");
		std::string str(temp);
		sigma = std::stof(str);
	}

	if(cmdOptionExists(argv, argv + argc, "--tau"))
	{
		char* temp = getCmdOption(argv, argv + argc, "--tau");
		std::string str(temp);
		tau = std::stof(str);
	}

	if(cmdOptionExists(argv, argv + argc, "--ratio"))
	{
		char* temp = getCmdOption(argv, argv + argc, "--ratio");
		std::string str(temp);
		ratio = std::stof(str);
	}

	printf("Input arguments are: R %f, sigma %f, tau %f.\n", ratio, sigma, tau);
	float tau2 = tau * tau;
	float sigma3 = 3 * sigma;

	// load & preprocess image
	int width;
	int height;
	int bpp;
	int depth = 3;
	unsigned char * image = stbi_load(input, &width, &height, &bpp, 3);
	size_t size = width * height;
	float* img;
	unsigned char* offset;
	img = (float*) malloc(size * depth * sizeof(float));
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			offset = image + (i * width + j) * depth;
			for(int k = 0; k < depth; k++){
				// one pixel contains 3 8-bit units, each unit is an ascii char.
				img[k * width * height + i * width + j] = (float) (offset[k] - 0); // numerate?
			}
		}
	}
	printf("image size: width %d height %d\n", width, height);

	// malloc space for host variables
	float *parent, *dist, *enrg;
	parent = (float*) calloc(size, sizeof(float)); // set 0 ???
	dist = (float*) calloc(size, sizeof(float));
	enrg = (float*) calloc(size, sizeof(float));

	// quick shift with shared memory on gpu

	getImg("input_check.png", img, width, height, depth);

	/*--------------------------------------------------------------------------*/
	// quick shift on gpu with texture;
	/*
	struct timeval gpu1, gpu2;
	gettimeofday(&gpu1, 0);
	quickshift(img, width, height, depth, sigma3, tau2, ratio, parent, dist);
	gettimeofday(&gpu2, 0);
	double gputime = (1000000.0 * (gpu2.tv_sec - gpu1.tv_sec) + gpu2.tv_usec - gpu1.tv_usec) / 1000.0;
	printf("img seg time of gpu is %f.\n", gputime);

	// gen img for gpu quick shift
	bool gpu = true;
	getOutput(img, parent, width, height, depth, gpu, sigma, tau, ratio);
	*/

	// quick shift on gpu;
	struct timeval gpu1, gpu2;
	gettimeofday(&gpu1, 0);
	quickshift_shared(img, width, height, depth, sigma3, tau2, ratio, parent);
	gettimeofday(&gpu2, 0);
	double gputime = (1000000.0 * (gpu2.tv_sec - gpu1.tv_sec) +
										gpu2.tv_usec - gpu1.tv_usec) / 1000.0;
	printf("img seg time of gpu is %f.\n", gputime);

	// gen img for gpu quick shift
	bool gpu = true;
	getOutput(img, parent, width, height, depth, gpu, sigma, tau, ratio);

	// quick shift with shared mem on gpu
	float* red = (float*) malloc(size * sizeof(float));
	float* green = (float*) malloc(size * sizeof(float));
	float* blue = (float*) malloc(size * sizeof(float));
	memcpy(img, red, size * sizeof(float));
	memcpy(img + size, green, size * sizeof(float));
	memcpy(img + 2 * size, blue, size * sizeof(float));

	// quick shift on cpu
	struct timeval cpu1, cpu2;
	gettimeofday(&cpu1, 0);
	serialquickshift(img, enrg, width, height, depth, sigma3, tau2, ratio, parent, dist);
	gettimeofday(&cpu2, 0);
	double cputime = (1000000.0 * (cpu2.tv_sec - cpu1.tv_sec) +
										cpu2.tv_usec - cpu1.tv_usec) / 1000.0;
	printf("img seg time of cpu is %f.\n", cputime);

	printf("I'm here.");
	free(parent);
	free(dist);
	free(enrg);
	free(img);
	free(image);
}
