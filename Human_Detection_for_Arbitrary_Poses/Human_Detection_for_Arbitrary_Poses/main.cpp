#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "svm.h"

#define YUDO_CD 0.7
#define YUDO_FD 0.9

using namespace std;

struct svm_node *x;
int max_nr_attr = 64;

double FD_max_ans = 0.0;
int FD_max_XY = 0;

struct svm_model* CD;

struct svm_model* FD_16x44;
struct svm_model* FD_16x48;
struct svm_model* FD_16x52;
struct svm_model* FD_16x56;
struct svm_model* FD_16x60;
struct svm_model* FD_16x64;

struct svm_model* FD_20x36;
struct svm_model* FD_20x40;
struct svm_model* FD_20x56;
struct svm_model* FD_20x60;
struct svm_model* FD_20x64;

struct svm_model* FD_24x36;
struct svm_model* FD_24x40;
struct svm_model* FD_24x44;
struct svm_model* FD_24x48;
struct svm_model* FD_24x64;

struct svm_model* FD_28x40;
struct svm_model* FD_28x44;
struct svm_model* FD_28x48;
struct svm_model* FD_28x52;
struct svm_model* FD_28x56;

struct svm_model* FD_32x48;
struct svm_model* FD_32x52;
struct svm_model* FD_32x56;
struct svm_model* FD_32x60;
struct svm_model* FD_32x64;

struct svm_model* FD_36x52;
struct svm_model* FD_36x56;
struct svm_model* FD_36x60;
struct svm_model* FD_36x64;

struct svm_model* FD_40x56;
struct svm_model* FD_40x60;
struct svm_model* FD_40x64;

struct svm_model* FD_44x16;
struct svm_model* FD_44x64;

struct svm_model* FD_48x16;

struct svm_model* FD_52x16;
struct svm_model* FD_52x20;

struct svm_model* FD_56x16;
struct svm_model* FD_56x20;

struct svm_model* FD_60x16;
struct svm_model* FD_60x20;

struct svm_model* FD_64x16;
struct svm_model* FD_64x20;
struct svm_model* FD_64x24;

//static char *line = NULL;
static int max_line_len;

class Detect_Place {
public:
	int C_x;
	int C_y;
	float C_yudo;

	int F_width;
	int F_height;
	float F_yudo;

	int territory_num;

	int ratio_num;

	Detect_Place() {
		C_x = C_y = -1;
		C_yudo = 0.0;

		F_width = F_height = -1;
		F_yudo = 0.0;

		territory_num = -1;

		ratio_num = -1;
	}

};

void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}

double predict(float *hog_vector, int hog_dim, svm_model* Detector)
{
	int svm_type = svm_get_svm_type(Detector);
	int nr_class = svm_get_nr_class(Detector);
	double *prob_estimates = NULL;
	int j;

	int *labels = (int *)malloc(nr_class * sizeof(int));
	svm_get_labels(Detector, labels);
	prob_estimates = (double *)malloc(nr_class * sizeof(double));
	free(labels);


	max_line_len = 1024;
	//	line = (char *)malloc(max_line_len*sizeof(char));
	x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
	int i;
	double target_label, predict_label;
	char *idx, *val, *label, *endptr;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0


							 //	for (i = 0; hog_vector[i] != NULL; i++)
	for (i = 0; i < hog_dim; i++)
	{
		//		clog << i << endl;
		if (i >= max_nr_attr - 1)	// need one more for index = -1
		{
			max_nr_attr *= 2;
			x = (struct svm_node *) realloc(x, max_nr_attr * sizeof(struct svm_node));
		}

		//			clog << i << endl;

		errno = 0;
		x[i].index = i;
		inst_max_index = x[i].index;

		errno = 0;
		x[i].value = hog_vector[i];
		//		cout << i << ":" << hog_vector[i] << endl;
	}
	x[i].index = -1;


	predict_label = svm_predict_probability(Detector, x, prob_estimates);
	//	if (prob_estimates[0] >= YUDO_CD)
	//		printf(" %f\n", prob_estimates[0]);

	free(x);
	//	free(line);

	return prob_estimates[0];
}

int minimum(int a, int b) {
	if (a<b) {
		return a;
	}
	return b;
}

void get_HOG(cv::Mat im, float* hog_vector) {
	int cell_size = 6;
	int rot_res = 9;
	int block_size = 3;
	int x, y, i, j, k, m, n, count;
	float dx, dy;
	float ***hist, *vec_tmp;
	float norm;
	CvMat *mag = NULL, *theta = NULL;
	//	FILE *hog_hist;

	//	fopen_s(&hog_hist,"d_im_hog.txt", "w");
	//	fopen_s(&hog_hist, "d_im_hog.bin", "w");
	//	fprintf(hog_hist, "%c ", '1');
	int counter = 1;

	mag = cvCreateMat(im.rows, im.cols, CV_32F);
	theta = cvCreateMat(im.rows, im.cols, CV_32F);
	for (y = 0; y<im.rows; y++) {
		for (x = 0; x<im.cols; x++) {
			if (x == 0 || x == im.cols - 1 || y == 0 || y == im.rows - 1) {
				cvmSet(mag, y, x, 0.0);
				cvmSet(theta, y, x, 0.0);
			}
			else {
				dx = double((uchar)im.data[y*im.step + x + 1]) - double((uchar)im.data[y*im.step + x - 1]);
				dy = double((uchar)im.data[(y + 1)*im.step + x]) - double((uchar)im.data[(y - 1)*im.step + x]);
				cvmSet(mag, y, x, sqrt(dx*dx + dy * dy));
				cvmSet(theta, y, x, atan(dy / (dx + 0.01)));
			}
		}
	}

	// histogram generation for each cell
	hist = (float***)malloc(sizeof(float**) * (int)ceil((float)im.rows / (float)cell_size));
	if (hist == NULL) exit(1);
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		hist[i] = (float**)malloc(sizeof(float*)*(int)ceil((float)im.cols / (float)cell_size));
		if (hist[i] == NULL) exit(1);
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++) {
			hist[i][j] = (float *)malloc(sizeof(float)*rot_res);
			if (hist[i][j] == NULL) exit(1);
		}
	}
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++) {
			for (k = 0; k<rot_res; k++) {
				hist[i][j][k] = 0.0;
			}
		}
	}
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++) {
			for (m = i * cell_size; m<minimum((i + 1)*cell_size, im.rows); m++) {
				for (n = j * cell_size; n<minimum((j + 1)*cell_size, im.cols); n++) {
					hist[i][j][(int)floor((cvmGet(theta, m, n) + CV_PI / 2)*rot_res / CV_PI)] += cvmGet(mag, m, n);
				}
			}
		}
	}

	// normalization for each block & generate vector
	vec_tmp = (float *)malloc(sizeof(float)*block_size*block_size*rot_res);
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size) - (block_size - 1); i++) {
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size) - (block_size - 1); j++) {
			count = 0;
			norm = 0.0;
			for (m = i; m<i + block_size; m++) {
				for (n = j; n<j + block_size; n++) {
					for (k = 0; k<rot_res; k++) {
						vec_tmp[count++] = hist[m][n][k];
						norm += hist[m][n][k] * hist[m][n][k];
					}
				}
			}
			for (count = 0; count<block_size*block_size*rot_res; count++) {
				vec_tmp[count] = vec_tmp[count] / (sqrt(norm + 1));
				if (vec_tmp[count]>0.2) vec_tmp[count] = 0.2;
				//		fprintf(hog_hist, "%d:%.4f ", counter, vec_tmp[count]);
				hog_vector[counter] = vec_tmp[count];
				//		cout << counter << ":" << hog_vector[counter] << endl;
				//		printf("%d:%.4f ",counter, vec_tmp[count]);
				counter++;
			}
		}
	}
	//	printf("\n");
	//	fprintf(hog_hist, "\n");
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j <(int)ceil((float)im.cols / (float)cell_size); j++) {
			free(hist[i][j]);
		}
		free(hist[i]);
	}
	free(hist);
	cvReleaseMat(&mag);
	cvReleaseMat(&theta);
	//	fclose(hog_hist);
}

cv::Mat draw_rectangle(cv::Mat ans_im, int x, int y, int width, int height, int r, int g, int b) {
	rectangle(ans_im, cvPoint(x, y), cvPoint(x + width, y + height), CV_RGB(r, g, b), 1);
	return ans_im;
}

int dimension(int x, int y) {

	return (int)(81 * ((int)ceil((float)x / 6) - 2) * ((int)ceil((float)y / 6) - 2));

	if (x % 6 == 0) {
		return 81 * (x / 6 - 2) * (y / 6 - 1);
	}
	if (y % 6 == 0) {
		return 81 * (x / 6 - 1) * (y / 6 - 2);
	}
	if (x % 6 == 0 && y % 6 == 0) {
		return 81 * (x / 6 - 2) * (y / 6 - 2);
	}
	else
		return 81 * (x / 6 - 1) * (y / 6 - 1);
}

void FD_predict(int width, int height, cv::Mat FD_img, svm_model* Detector) {
	cv::Mat FD_cut_im(FD_img, cv::Rect(32 - width / 2, 32 - height / 2, width, height));
	//	cv::Mat FD_max_img = FD_cut_im;
	//	cvtColor(FD_cut_im, FD_cut_im, CV_RGB2GRAY);
	int hog_dim = dimension(FD_cut_im.cols, FD_cut_im.rows);
	float FD_vector[6562];	//�e������HOG���i�[
	get_HOG(FD_cut_im, FD_vector);	//HOG�̎擾

	double ans = predict(FD_vector, hog_dim, Detector);

	//	cout << "x,y:" << width << "," << height << "=" << ans << endl;

	if (FD_max_ans < ans && YUDO_FD < ans) {
		FD_max_ans = ans;
		FD_max_XY = width * 100 + height;
		//	cv::imwrite("result_FD.bmp", FD_cut_im);
	}
}

int main(int argc, char** argv) {
	//�ϐ��錾
	//	int x, y;

	int file_num = 1;

	int count = 0;
	int hog_dim;

	//CoarseDetector�̎�荞��
	if ((CD = svm_load_model("C:/model_file/CD_from_demo.model")) == 0)exit(1);

	if ((FD_16x44 = svm_load_model("C:/model_file/FD_16x44.model")) == 0)exit(1);

	if ((FD_16x48 = svm_load_model("C:/model_file/FD_16x48.model")) == 0)exit(1);
	if ((FD_16x52 = svm_load_model("C:/model_file/FD_16x52.model")) == 0)exit(1);
	if ((FD_16x56 = svm_load_model("C:/model_file/FD_16x56.model")) == 0)exit(1);
	if ((FD_16x60 = svm_load_model("C:/model_file/FD_16x60.model")) == 0)exit(1);
	if ((FD_16x64 = svm_load_model("C:/model_file/FD_16x64.model")) == 0)exit(1);

	if ((FD_20x36 = svm_load_model("C:/model_file/FD_20x36.model")) == 0)exit(1);
	if ((FD_20x40 = svm_load_model("C:/model_file/FD_20x40.model")) == 0)exit(1);
	if ((FD_20x56 = svm_load_model("C:/model_file/FD_20x56.model")) == 0)exit(1);
	if ((FD_20x60 = svm_load_model("C:/model_file/FD_20x60.model")) == 0)exit(1);
	if ((FD_20x64 = svm_load_model("C:/model_file/FD_20x64.model")) == 0)exit(1);

	if ((FD_24x36 = svm_load_model("C:/model_file/FD_24x36.model")) == 0)exit(1);
	if ((FD_24x40 = svm_load_model("C:/model_file/FD_24x40.model")) == 0)exit(1);
	if ((FD_24x44 = svm_load_model("C:/model_file/FD_24x44.model")) == 0)exit(1);
	if ((FD_24x48 = svm_load_model("C:/model_file/FD_24x48.model")) == 0)exit(1);
	if ((FD_24x64 = svm_load_model("C:/model_file/FD_24x64.model")) == 0)exit(1);

	if ((FD_28x40 = svm_load_model("C:/model_file/FD_28x40.model")) == 0)exit(1);
	if ((FD_28x44 = svm_load_model("C:/model_file/FD_28x44.model")) == 0)exit(1);
	if ((FD_28x48 = svm_load_model("C:/model_file/FD_28x48.model")) == 0)exit(1);
	if ((FD_28x52 = svm_load_model("C:/model_file/FD_28x52.model")) == 0)exit(1);
	if ((FD_28x56 = svm_load_model("C:/model_file/FD_28x56.model")) == 0)exit(1);


	if ((FD_32x48 = svm_load_model("C:/model_file/FD_32x48.model")) == 0)exit(1);
	if ((FD_32x52 = svm_load_model("C:/model_file/FD_32x52.model")) == 0)exit(1);
	if ((FD_32x56 = svm_load_model("C:/model_file/FD_32x56.model")) == 0)exit(1);
	if ((FD_32x60 = svm_load_model("C:/model_file/FD_32x60.model")) == 0)exit(1);
	if ((FD_32x64 = svm_load_model("C:/model_file/FD_32x64.model")) == 0)exit(1);

	if ((FD_36x52 = svm_load_model("C:/model_file/FD_36x52.model")) == 0)exit(1);
	if ((FD_36x56 = svm_load_model("C:/model_file/FD_36x56.model")) == 0)exit(1);
	if ((FD_36x60 = svm_load_model("C:/model_file/FD_36x60.model")) == 0)exit(1);
	if ((FD_36x64 = svm_load_model("C:/model_file/FD_36x64.model")) == 0)exit(1);

	if ((FD_40x64 = svm_load_model("C:/model_file/FD_40x64.model")) == 0)exit(1);
	if ((FD_40x56 = svm_load_model("C:/model_file/FD_40x56.model")) == 0)exit(1);
	if ((FD_40x60 = svm_load_model("C:/model_file/FD_40x60.model")) == 0)exit(1);

	if ((FD_44x16 = svm_load_model("C:/model_file/FD_44x16.model")) == 0)exit(1);
	if ((FD_44x64 = svm_load_model("C:/model_file/FD_44x64.model")) == 0)exit(1);

	if ((FD_48x16 = svm_load_model("C:/model_file/FD_48x16.model")) == 0)exit(1);

	if ((FD_52x16 = svm_load_model("C:/model_file/FD_52x16.model")) == 0)exit(1);
	if ((FD_52x20 = svm_load_model("C:/model_file/FD_52x20.model")) == 0)exit(1);

	if ((FD_56x16 = svm_load_model("C:/model_file/FD_56x16.model")) == 0)exit(1);
	if ((FD_56x20 = svm_load_model("C:/model_file/FD_56x20.model")) == 0)exit(1);

	if ((FD_60x16 = svm_load_model("C:/model_file/FD_60x16.model")) == 0)exit(1);
	if ((FD_60x20 = svm_load_model("C:/model_file/FD_60x20.model")) == 0)exit(1);

	if ((FD_64x16 = svm_load_model("C:/model_file/FD_64x16.model")) == 0)exit(1);
	if ((FD_64x20 = svm_load_model("C:/model_file/FD_64x20.model")) == 0)exit(1);
	if ((FD_64x24 = svm_load_model("C:/model_file/FD_64x24.model")) == 0)exit(1);

	//�e�X�g�摜�t�@�C���ꗗ�������ǂݍ���
	char test_name[1024], result_name[1024];
	FILE *test_data, *result_data, *result_text;
	fopen_s(&test_data, "C:/photo/train_data_from_demo/before_normalize/list_origin.txt", "r");
	fopen_s(&result_data, "C:/photo/train_data_from_demo/result_data/list_result.txt", "r");

	while (fgets(test_name, 256, test_data) != NULL && fgets(result_name, 256, result_data) != NULL) {
		fopen_s(&result_text, "C:/photo/train_data_from_demo/result_data/result_text.txt", "a");

		string name_tes = test_name;
		char new_test_name[1024];
		for (int i = 0; i < name_tes.length() - 1; i++) {
			new_test_name[i] = test_name[i];
			new_test_name[i + 1] = '\0';
		}

		string name_res = result_name;
		char new_result_name[1024];
		for (int i = 0; i < name_res.length() - 1; i++) {
			new_result_name[i] = result_name[i];
			new_result_name[i + 1] = '\0';
		}

		count = 0;

		//�摜�̎�荞��
		cv::Mat ans_img_CF = cv::imread(new_test_name, 1);	//���o����摜
		cv::Mat res_bin = cv::Mat::zeros(ans_img_CF.rows, ans_img_CF.cols, CV_8UC3);
		
		//���U���g�t�@�C���ɉ摜�t�@�C��������������
		fprintf_s(result_text, new_result_name);

		cout << file_num << ":" << new_test_name << endl;
		file_num++;

		//Detect_Place�I�u�W�F�N�g�̍쐬
		Detect_Place detect[100];

		//Coarse Detector�ɂ��l�����o
		cv::Mat CD_img[100];
		
		float normalize_num[15] = { 68,96,128,160,192,224,-1 };

		for (int img_size = 0; normalize_num[img_size] != -1; img_size++) {
			cv::Mat img;			//���o��`�������{���摜
			cvtColor(ans_img_CF, img, CV_RGB2GRAY);
			cv::resize(img, img, cv::Size(), normalize_num[img_size] / img.rows, normalize_num[img_size] / img.rows, CV_INTER_LINEAR);
			for (int y = 2; (y + 68) <= img.rows; y += 4) {
				for (int x = 2; (x + 68) <= img.cols; x += 4) {
					cv::Mat d_im(img, cv::Rect(x, y, 64, 64));
					hog_dim = dimension(d_im.cols, d_im.rows);
					float hog_vector[6562];							//�e������HOG���i�[
					get_HOG(d_im, hog_vector);	//HOG�̎擾
					double ans = predict(hog_vector, hog_dim, CD);	//�ޓx�̎Z�o
					if (ans >= YUDO_CD) {//�ޓx����l������l�����̔��f
						detect[count].C_yudo = ans;
						detect[count].C_x = x;
						detect[count].C_y = y;
						detect[count].ratio_num = img_size;
						CD_img[count] = img.clone();
						CD_img[count] = CD_img[count](cv::Rect(x - 2, y - 2, 68, 68));
						count++;
					}
				}
			}
		}
		//	cv::imwrite("result_CD.bmp", ans_img_CD);

		//Fine Detector�ɂ�錟�o
		for (int i = 0; i < count; i++) {
			float zure_yudo[25];
			int zure_count = 0;
			FD_max_ans = 0.0;
			FD_max_XY = 0;

			for (int a = 0; a <= 4; a++) {
				for (int b = 0; b <= 4; b++) {

					cv::Mat FD_img = CD_img[i](cv::Rect(a, b, 64, 64));

					FD_predict(16, 44, FD_img, FD_16x44);
					FD_predict(16, 48, FD_img, FD_16x48);
					FD_predict(16, 52, FD_img, FD_16x52);
					FD_predict(16, 56, FD_img, FD_16x56);
					FD_predict(16, 60, FD_img, FD_16x60);
					FD_predict(16, 64, FD_img, FD_16x64);

					FD_predict(20, 36, FD_img, FD_20x36);
					FD_predict(20, 40, FD_img, FD_20x40);
					FD_predict(20, 56, FD_img, FD_20x56);
					FD_predict(20, 60, FD_img, FD_20x60);
					FD_predict(20, 64, FD_img, FD_20x64);

					FD_predict(24, 36, FD_img, FD_24x36);
					FD_predict(24, 40, FD_img, FD_24x40);
					FD_predict(24, 44, FD_img, FD_24x44);
					FD_predict(24, 48, FD_img, FD_24x48);
					FD_predict(24, 64, FD_img, FD_24x64);

					FD_predict(28, 40, FD_img, FD_28x40);
					FD_predict(28, 44, FD_img, FD_28x44);
					FD_predict(28, 48, FD_img, FD_28x48);
					FD_predict(28, 52, FD_img, FD_28x52);
					FD_predict(28, 56, FD_img, FD_28x56);

					FD_predict(32, 48, FD_img, FD_32x48);
					FD_predict(32, 52, FD_img, FD_32x52);
					FD_predict(32, 56, FD_img, FD_32x56);
					FD_predict(32, 60, FD_img, FD_32x60);
					FD_predict(32, 64, FD_img, FD_32x64);

					FD_predict(36, 52, FD_img, FD_36x52);
					FD_predict(36, 56, FD_img, FD_36x56);
					FD_predict(36, 60, FD_img, FD_36x60);
					FD_predict(36, 64, FD_img, FD_36x64);

					FD_predict(40, 56, FD_img, FD_40x56);
					FD_predict(40, 60, FD_img, FD_40x60);
					FD_predict(40, 64, FD_img, FD_40x64);

					FD_predict(44, 16, FD_img, FD_44x16);
					FD_predict(44, 64, FD_img, FD_44x64);

					FD_predict(48, 16, FD_img, FD_48x16);

					FD_predict(52, 16, FD_img, FD_52x16);
					FD_predict(52, 20, FD_img, FD_52x20);

					FD_predict(56, 16, FD_img, FD_56x16);
					FD_predict(56, 20, FD_img, FD_56x20);

					FD_predict(60, 16, FD_img, FD_60x16);
					FD_predict(60, 20, FD_img, FD_60x20);

					FD_predict(64, 16, FD_img, FD_64x16);
					FD_predict(64, 20, FD_img, FD_64x20);
					FD_predict(64, 24, FD_img, FD_64x24);

					if (FD_max_ans > YUDO_FD && FD_max_ans > detect[i].F_yudo) {
						detect[i].F_yudo = FD_max_ans;
						detect[i].F_width = FD_max_XY / 100;
						detect[i].F_height = FD_max_XY % 100;
						zure_count = a * 10 + b;
					}
				}
			}

			if (zure_count != 0) {
				detect[i].C_x += (zure_count / 10) - 2;
				detect[i].C_y += (zure_count % 10) - 2;
			}
		}

		//�̈�̓���
		int t_num = 0;
		for (int n = 0; detect[n].C_yudo != 0; n++) {
			if (detect[n].F_yudo == 0) continue;
			if (detect[n].territory_num == -1) {
				t_num++;
				detect[n].territory_num = t_num;
			}
			for (int m = n + 1; detect[m].C_yudo != 0; m++) {
				if (detect[m].F_yudo == 0) continue;

				if ((detect[n].C_x + 32) - 100 <= (detect[m].C_x + 32)
					&& (detect[m].C_x + 32) <= (detect[n].C_x + 32) + 100
					&&
					(detect[n].C_y + 32) - 100 <= (detect[m].C_y + 32)
					&& (detect[m].C_y + 32) <= (detect[n].C_y + 32) + 100) {

					detect[m].territory_num = detect[n].territory_num;
				}
			}

		}
		//����̈悲�ƂɌ��o���ʂ̕\��
		for (int i = 1; i <= t_num; i++) {
			int final_num = 0;
			float fyudo = 0;
			for (int k = 0; detect[k].C_yudo != 0; k++) {
				if (detect[k].territory_num == i && detect[k].F_yudo > fyudo) {
					final_num = k;
					fyudo = detect[k].F_yudo;
				}
			}
			int C_x1 = detect[final_num].C_x * 480 / normalize_num[detect[final_num].ratio_num];
			int C_y1 = detect[final_num].C_y * 480 / normalize_num[detect[final_num].ratio_num];
			int C_width = 64 * 480 / normalize_num[detect[final_num].ratio_num];
			int C_height = 64 * 480 / normalize_num[detect[final_num].ratio_num];

			int F_x1 = (detect[final_num].C_x + 32 - detect[final_num].F_width / 2) * 480 / normalize_num[detect[final_num].ratio_num];
			int F_y1 = (detect[final_num].C_y + 32 - detect[final_num].F_height / 2) * 480 / normalize_num[detect[final_num].ratio_num];
			int F_width = detect[final_num].F_width * 480 / normalize_num[detect[final_num].ratio_num];
			int F_height = detect[final_num].F_height * 480 / normalize_num[detect[final_num].ratio_num];

			//��`�\��
			//CD�̋�`
			ans_img_CF = draw_rectangle(ans_img_CF, C_x1, C_y1, C_width, C_height, 255, 0, 0);
			//FD�̋�`
			ans_img_CF = draw_rectangle(ans_img_CF,F_x1, F_y1,F_width,F_height, 0, 255, 0);
			//FD���ʂ��e�L�X�g�t�@�C���ɕۑ�
			fprintf_s(result_text, " , %d, %d, %d, %d", F_x1, F_y1, F_x1 + F_width, F_y1 + F_height);

			cout << C_x1 << endl;
			cout << C_y1 << endl;
			cout << C_width << endl;
			cout << F_x1 << endl;
			cout << F_y1 << endl;
			cout << F_width << endl;
			cout << F_height << endl;

//			for (int n = F_y1; n < F_y1 + F_height; n++) {
//				for (int m = F_x1; m < F_x1 + F_width; m++) {
			for(int n=165;n<365;n++){
				for(int m=215;m<335;m++){
					res_bin.at<cv::Vec3b>(n, m) = cv::Vec3b(255, 255, 255);
				}
			}

		}
		//�e�L�X�g�t�@�C�����s
		fprintf_s(result_text, "\n");

		cv::cvtColor(res_bin, res_bin, CV_RGB2GRAY);
		cv::imwrite("res_bin.jpg", res_bin);
		
		//�摜�̕ۑ�(���o���ł��Ă��Ă����Ȃ��Ă��ۑ�)
		cv::imwrite(new_result_name, ans_img_CF);

		fclose(result_text);
	}
	fclose(test_data);
	fclose(result_data);

	return 0;
}
