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

#define YUDO_CD 0.0
#define YUDO_FD 0.0

using namespace std;

struct svm_node *x;
int max_nr_attr = 64;

double FD_max_ans = 0.0;
int FD_max_XY = 0;

struct svm_model* CD;

struct svm_model* FD_24x72;
struct svm_model* FD_24x80;
struct svm_model* FD_24x88;

struct svm_model* FD_32x88;
struct svm_model* FD_32x96;
struct svm_model* FD_32x104;
struct svm_model* FD_32x112;
struct svm_model* FD_32x120;

struct svm_model* FD_40x72;
struct svm_model* FD_40x80;
struct svm_model* FD_40x112;
struct svm_model* FD_40x120;
struct svm_model* FD_40x128;

struct svm_model* FD_48x80;
struct svm_model* FD_48x88;
struct svm_model* FD_48x96;
struct svm_model* FD_48x128;

struct svm_model* FD_56x96;
struct svm_model* FD_56x104;
struct svm_model* FD_56x112;

struct svm_model* FD_64x104;
struct svm_model* FD_64x112;
struct svm_model* FD_64x120;
struct svm_model* FD_64x128;

struct svm_model* FD_72x16;
struct svm_model* FD_72x120;
struct svm_model* FD_72x128;

struct svm_model* FD_80x16;
struct svm_model* FD_80x24;
struct svm_model* FD_80x128;

struct svm_model* FD_88x24;
struct svm_model* FD_88x32;

struct svm_model* FD_96x24;
struct svm_model* FD_96x32;

struct svm_model* FD_104x24;
struct svm_model* FD_104x32;
struct svm_model* FD_104x40;

struct svm_model* FD_112x24;
struct svm_model* FD_112x32;
struct svm_model* FD_112x40;

struct svm_model* FD_120x24;
struct svm_model* FD_120x32;
struct svm_model* FD_120x40;

struct svm_model* FD_128x32;
struct svm_model* FD_128x40;
struct svm_model* FD_128x48;

//static char *line = NULL;
static int max_line_len;

class Detect_Place {
public:
	int C_x;
	int C_y;
	int C_width;
	int C_height;
	float C_yudo;

	int F_x;
	int F_y;
	int F_width;
	int F_height;
	float F_yudo;

	int territory_num;

	int ratio_num;

	Detect_Place() {
		C_x = C_y = -1;
		C_width = C_height = -1;
		C_yudo = 0.0;

		F_x = F_y = -1;
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
	cv::Mat FD_cut_im(FD_img, cv::Rect(64 - width / 2, 64 - height / 2, width, height));
	int hog_dim = dimension(FD_cut_im.cols, FD_cut_im.rows);
	float FD_vector[20000];	//各次元のHOGを格納
	get_HOG(FD_cut_im, FD_vector);	//HOGの取得

	double ans = predict(FD_vector, hog_dim, Detector);

	if (FD_max_ans < ans && YUDO_FD < ans) {
		FD_max_ans = ans;
		FD_max_XY = width * 1000 + height;
		
		//	cv::imwrite("result_FD.bmp", FD_cut_im);
	}
}

void detect_1() {
	//変数宣言
	//	int x, y;

	int file_num = 1;

	int count = 0;
	int hog_dim;

	//CoarseDetectorの取り込み
	if ((CD = svm_load_model("C:/model_file/pre_model/CD_123.model")) == 0)exit(1);

	if ((FD_24x72 = svm_load_model("C:/model_file/pre_model/FD_24x72.model")) == 0)exit(1);
	if ((FD_24x80 = svm_load_model("C:/model_file/pre_model/FD_24x80.model")) == 0)exit(1);
	if ((FD_24x88 = svm_load_model("C:/model_file/pre_model/FD_24x88.model")) == 0)exit(1);

	if ((FD_32x88 = svm_load_model("C:/model_file/pre_model/FD_32x88.model")) == 0)exit(1);
	if ((FD_32x96 = svm_load_model("C:/model_file/pre_model/FD_32x96.model")) == 0)exit(1);
	if ((FD_32x104 = svm_load_model("C:/model_file/pre_model/FD_32x104.model")) == 0)exit(1);
	if ((FD_32x112 = svm_load_model("C:/model_file/pre_model/FD_32x112.model")) == 0)exit(1);
	if ((FD_32x120 = svm_load_model("C:/model_file/pre_model/FD_32x120.model")) == 0)exit(1);

	if ((FD_40x72 = svm_load_model("C:/model_file/pre_model/FD_40x72.model")) == 0)exit(1);
	if ((FD_40x80 = svm_load_model("C:/model_file/pre_model/FD_40x80.model")) == 0)exit(1);
	if ((FD_40x112 = svm_load_model("C:/model_file/pre_model/FD_40x112.model")) == 0)exit(1);
	if ((FD_40x120 = svm_load_model("C:/model_file/pre_model/FD_40x120.model")) == 0)exit(1);
	if ((FD_40x128 = svm_load_model("C:/model_file/pre_model/FD_40x128.model")) == 0)exit(1);

	if ((FD_48x80 = svm_load_model("C:/model_file/pre_model/FD_48x80.model")) == 0)exit(1);
	if ((FD_48x88 = svm_load_model("C:/model_file/pre_model/FD_48x88.model")) == 0)exit(1);
	if ((FD_48x96 = svm_load_model("C:/model_file/pre_model/FD_48x96.model")) == 0)exit(1);
	if ((FD_48x128 = svm_load_model("C:/model_file/pre_model/FD_48x128.model")) == 0)exit(1);

	if ((FD_56x96 = svm_load_model("C:/model_file/pre_model/FD_56x96.model")) == 0)exit(1);
	if ((FD_56x104 = svm_load_model("C:/model_file/pre_model/FD_56x104.model")) == 0)exit(1);
	if ((FD_56x112 = svm_load_model("C:/model_file/pre_model/FD_56x112.model")) == 0)exit(1);

	if ((FD_64x104 = svm_load_model("C:/model_file/pre_model/FD_64x104.model")) == 0)exit(1);
	if ((FD_64x112 = svm_load_model("C:/model_file/pre_model/FD_64x112.model")) == 0)exit(1);
	if ((FD_64x120 = svm_load_model("C:/model_file/pre_model/FD_64x120.model")) == 0)exit(1);
	if ((FD_64x128 = svm_load_model("C:/model_file/pre_model/FD_64x128.model")) == 0)exit(1);

	if ((FD_72x16 = svm_load_model("C:/model_file/pre_model/FD_72x16.model")) == 0)exit(1);
	if ((FD_72x120 = svm_load_model("C:/model_file/pre_model/FD_72x120.model")) == 0)exit(1);
	if ((FD_72x128 = svm_load_model("C:/model_file/pre_model/FD_72x128.model")) == 0)exit(1);

	if ((FD_80x16 = svm_load_model("C:/model_file/pre_model/FD_80x16.model")) == 0)exit(1);
	if ((FD_80x24 = svm_load_model("C:/model_file/pre_model/FD_80x24.model")) == 0)exit(1);
	if ((FD_80x128 = svm_load_model("C:/model_file/pre_model/FD_80x128.model")) == 0)exit(1);

	if ((FD_88x24 = svm_load_model("C:/model_file/pre_model/FD_88x24.model")) == 0)exit(1);
	if ((FD_88x32 = svm_load_model("C:/model_file/pre_model/FD_88x32.model")) == 0)exit(1);

	if ((FD_96x24 = svm_load_model("C:/model_file/pre_model/FD_96x24.model")) == 0)exit(1);
	if ((FD_96x32 = svm_load_model("C:/model_file/pre_model/FD_96x32.model")) == 0)exit(1);

	if ((FD_104x24 = svm_load_model("C:/model_file/pre_model/FD_104x24.model")) == 0)exit(1);
	if ((FD_104x32 = svm_load_model("C:/model_file/pre_model/FD_104x32.model")) == 0)exit(1);
	if ((FD_104x40 = svm_load_model("C:/model_file/pre_model/FD_104x40.model")) == 0)exit(1);

	if ((FD_112x24 = svm_load_model("C:/model_file/pre_model/FD_112x24.model")) == 0)exit(1);
	if ((FD_112x32 = svm_load_model("C:/model_file/pre_model/FD_112x32.model")) == 0)exit(1);
	if ((FD_112x40 = svm_load_model("C:/model_file/pre_model/FD_112x40.model")) == 0)exit(1);

	if ((FD_120x24 = svm_load_model("C:/model_file/pre_model/FD_120x24.model")) == 0)exit(1);
	if ((FD_120x32 = svm_load_model("C:/model_file/pre_model/FD_120x32.model")) == 0)exit(1);
	if ((FD_120x40 = svm_load_model("C:/model_file/pre_model/FD_120x40.model")) == 0)exit(1);

	if ((FD_128x32 = svm_load_model("C:/model_file/pre_model/FD_128x32.model")) == 0)exit(1);
	if ((FD_128x40 = svm_load_model("C:/model_file/pre_model/FD_128x40.model")) == 0)exit(1);
	if ((FD_128x48 = svm_load_model("C:/model_file/pre_model/FD_128x48.model")) == 0)exit(1);



	//テスト画像ファイル一覧メモ帳読み込み
	char test_name[1024], result_name[1024];
	FILE *test_data, *result_data;
	if (fopen_s(&test_data, "c:/photo/test_list_2.txt", "r") != 0) {
		cout << "missing" << endl;
		return ;
	}

	while (fgets(test_name, 256, test_data) != NULL) {
		string name_tes = test_name;
		char new_test_name[1024];
		for (int i = 0; i < name_tes.length() - 1; i++) {
			new_test_name[i] = test_name[i];
			new_test_name[i + 1] = '\0';
		}
		count = 0;

		char test_path[1024] = "C:/photo/test_data_from_demo/test_data/";
		strcat_s(test_path, new_test_name);

		for (int i = 0; i < 1024; i++) {
			if (new_test_name[i] == 'b') {
				new_test_name[i] = 't';
				new_test_name[i + 1] = 'x';
				new_test_name[i + 2] = 't';
				new_test_name[i + 3] = '\0';
				break;
			}
			else new_test_name[i] = new_test_name[i];
		}
		char result_path[1024] = "result_data/";
		strcat_s(result_path, new_test_name);

		if (fopen_s(&result_data, result_path, "a") != 0) {
			cout << "missing 2" << endl;
			return ;
		}
		//画像の取り込み
		cv::Mat ans_img_CF = cv::imread(test_path, 1);	//検出する画像
														//		cv::Mat ans_img_CF = cv::imread("Sun_Nov_26_14_02_00_95.bmp", 1);	//検出する画像
		cv::Mat check_img = ans_img_CF.clone();

		cv::Mat img;			//検出矩形処理を施す画像
		cvtColor(ans_img_CF, img, CV_RGB2GRAY);

		cout << file_num << ":" << new_test_name << endl;
		file_num++;

		//Detect_Placeオブジェクトの作成
		Detect_Place detect[500];

		//Coarse Detectorによる人物検出
		//	cv::Mat CD_img[500];

		float normalize_num[15] = { 128, 160,192, 224, 256, 288, 320, -1 };

		for (int img_size = 0; normalize_num[img_size] != -1; img_size++) {
			//	cv::Mat img;			//検出矩形処理を施す画像
			//	cvtColor(ans_img_CF, img, CV_RGB2GRAY);
			cvtColor(ans_img_CF, img, CV_RGB2GRAY);
			cv::Mat ans_CD = img.clone();

			cv::resize(img, img, cv::Size(), normalize_num[img_size] / img.rows, normalize_num[img_size] / img.rows, CV_INTER_LINEAR);
			for (int y = 0; (y + 128) <= img.rows; y += 8) {
				for (int x = 0; (x + 128) <= img.cols; x += 8) {
					cv::Mat d_im(img, cv::Rect(x, y, 128, 128));
					cv::resize(d_im, d_im, cv::Size(), 96.0 / d_im.cols, 96.0 / d_im.rows);

					hog_dim = dimension(d_im.cols, d_im.rows);
					float hog_vector[20000];						//各次元のHOGを格納
					get_HOG(d_im, hog_vector);	//HOGの取得
					double ans = predict(hog_vector, hog_dim, CD);	//尤度の算出
					if (ans >= YUDO_CD) {//尤度から人物か非人物かの判断
										 //	cout << ans << endl;

						detect[count].C_yudo = ans;
						detect[count].C_x = x; //*480 / normalize_num[img_size];
						detect[count].C_y = y; //*480 / normalize_num[img_size];
						detect[count].C_width = 128 * 480 / normalize_num[img_size];
						detect[count].C_height = 128 * 480 / normalize_num[img_size];
						detect[count].ratio_num = img_size;
						//	if (x >= 2) {
						//		if ((x + 132) <= img.cols) {

						//		}
						//	CD_img[count] = img(cv::Rect(x - 2, y - 2, 132, 132));
						//	}
						//	cv::imshow("", CD_img[count]);
						//	cvWaitKey(10);
						//	CD_img[count] = CD_img[count](cv::Rect(x, y, 128, 128));
						int C_x1 = detect[count].C_x * 480 / normalize_num[img_size];
						int C_y1 = detect[count].C_y * 480 / normalize_num[img_size];
						check_img = draw_rectangle(check_img, C_x1, C_y1, detect[count].C_width, detect[count].C_height, 255, 0, 0);
						count++;
					}
				}
			}
		}

		//	cv::imshow("", check_img);
		//	cvWaitKey(0);

		//Fine Detectorによる検出
		for (int i = 0; i < count; i++) {
			float zure_yudo[25];
			int zure_count = 0;
			FD_max_ans = 0.0;
			FD_max_XY = 0;

			cvtColor(ans_img_CF, img, CV_RGB2GRAY);
			cv::resize(img, img, cv::Size(), normalize_num[detect[i].ratio_num] / img.rows, normalize_num[detect[i].ratio_num] / img.rows);

			for (int a = 0; a <= 4; a += 2) {
				for (int b = 0; b <= 4; b += 2) {
					cv::Mat FD_img;
					//	cv::Mat FD_img = CD_img[i](cv::Rect(a, b, 128, 128));
					//	if ((detect[i].C_x + (a - 2) * 480 / normalize_num[detect[i].ratio_num]) >= 0 && (detect->C_y + (b - 2) / normalize_num[detect[i].ratio_num]) >= 0 && 
					//		(detect[i].C_x + (a - 2 + 128) * 480 / normalize_num[detect[i].ratio_num]) <= img.cols && (detect[i].C_y + (b - 2 + 128) * 480 / normalize_num[detect[i].ratio_num]) <= img.rows) {
					if (detect[i].C_x + a - 2 >= 0 && detect[i].C_y + b - 2 >= 0 && detect[i].C_x + a - 2 + 128 <= img.cols && detect[i].C_y + b - 2 + 128 <= img.rows) {
						FD_img = img(cv::Rect(detect[i].C_x + a - 2, detect[i].C_y + b - 2, 128, 128));
					}
					else {
						continue;
					}
					//					cv::Mat FD_img = img(cv::Rect(detect[i].C_x + a - 2, detect[i].C_y + b - 2, 128, 128));
					FD_predict(24, 72, FD_img, FD_24x72);
					FD_predict(24, 80, FD_img, FD_24x80);
					FD_predict(24, 88, FD_img, FD_24x88);

					FD_predict(32, 88, FD_img, FD_32x88);
					FD_predict(32, 96, FD_img, FD_32x96);
					FD_predict(32, 104, FD_img, FD_32x104);
					FD_predict(32, 112, FD_img, FD_32x112);
					FD_predict(32, 120, FD_img, FD_32x120);

					FD_predict(40, 72, FD_img, FD_40x72);
					FD_predict(40, 80, FD_img, FD_40x80);
					FD_predict(40, 112, FD_img, FD_40x112);
					FD_predict(40, 120, FD_img, FD_40x120);
					FD_predict(40, 128, FD_img, FD_40x128);

					FD_predict(48, 80, FD_img, FD_48x80);
					FD_predict(48, 88, FD_img, FD_48x88);
					FD_predict(48, 96, FD_img, FD_48x96);
					FD_predict(48, 128, FD_img, FD_48x128);

					FD_predict(56, 96, FD_img, FD_56x96);
					FD_predict(56, 104, FD_img, FD_56x104);
					FD_predict(56, 112, FD_img, FD_56x112);

					FD_predict(64, 104, FD_img, FD_64x104);
					FD_predict(64, 112, FD_img, FD_64x112);
					FD_predict(64, 120, FD_img, FD_64x120);
					FD_predict(64, 128, FD_img, FD_64x128);

					FD_predict(72, 16, FD_img, FD_72x16);
					FD_predict(72, 120, FD_img, FD_72x120);
					FD_predict(72, 128, FD_img, FD_72x128);

					FD_predict(80, 16, FD_img, FD_80x16);
					FD_predict(80, 24, FD_img, FD_80x24);
					FD_predict(80, 128, FD_img, FD_80x128);

					FD_predict(88, 24, FD_img, FD_88x24);
					FD_predict(88, 32, FD_img, FD_88x32);

					FD_predict(96, 24, FD_img, FD_96x24);
					FD_predict(96, 32, FD_img, FD_96x32);

					FD_predict(104, 24, FD_img, FD_104x24);
					FD_predict(104, 32, FD_img, FD_104x32);
					FD_predict(104, 40, FD_img, FD_104x40);

					FD_predict(112, 24, FD_img, FD_112x24);
					FD_predict(112, 32, FD_img, FD_112x32);
					FD_predict(112, 40, FD_img, FD_112x40);

					FD_predict(120, 24, FD_img, FD_120x24);
					FD_predict(120, 32, FD_img, FD_120x32);
					FD_predict(120, 40, FD_img, FD_120x40);

					FD_predict(128, 32, FD_img, FD_128x32);
					FD_predict(128, 40, FD_img, FD_128x40);
					FD_predict(128, 48, FD_img, FD_128x48);


					if (FD_max_ans > YUDO_FD && FD_max_ans > detect[i].F_yudo) {
						detect[i].F_yudo = FD_max_ans;
						detect[i].F_width = FD_max_XY / 1000;
						detect[i].F_height = FD_max_XY % 1000;
						zure_count = a * 10 + b;
					}
				}
			}

			//	if (zure_count != 0) {
			if (FD_max_ans != 0) {
				detect[i].C_x += (zure_count / 10) - 2;
				detect[i].C_y += (zure_count % 10) - 2;

				int F_x1 = (detect[i].C_x + 64 - detect[i].F_width / 2) * 480 / normalize_num[detect[i].ratio_num];
				int F_y1 = (detect[i].C_y + 64 - detect[i].F_height / 2) * 480 / normalize_num[detect[i].ratio_num];
				int F_width = detect[i].F_width * 480 / normalize_num[detect[i].ratio_num];
				int F_height = detect[i].F_height * 480 / normalize_num[detect[i].ratio_num];

				detect[i].F_x = F_x1;
				detect[i].F_y = F_y1;
				detect[i].F_width = F_width;
				detect[i].F_height = F_height;
				detect[i].C_x = detect[i].C_x * 480 / normalize_num[detect[i].ratio_num];
				detect[i].C_y = detect[i].C_y * 480 / normalize_num[detect[i].ratio_num];

				check_img = draw_rectangle(check_img, detect[i].C_x, detect[i].C_y, detect[i].C_width, detect[i].C_height, 255, 0, 0);
				check_img = draw_rectangle(check_img, detect[i].F_x, detect[i].F_y, detect[i].F_width, detect[i].F_height, 0, 255, 0);
				cout << "FD:(" << F_x1 << "," << F_y1 << "),(" << F_width << "," << F_height << "):" << F_width * F_height << ", " << detect[i].F_yudo << endl;;
			}
		}

		//領域の統一
		int t_num = 0;
		for (int n = 0; detect[n].C_yudo != 0; n++) {
			if (detect[n].F_yudo == 0) continue;
			if (detect[n].territory_num == -1) {
				t_num++;
				detect[n].territory_num = t_num;
			}
			for (int m = n + 1; detect[m].C_yudo != 0; m++) {
				if (detect[m].F_yudo == 0) continue;

				if ((detect[n].C_x + detect[n].C_width / 2) - 100 <= (detect[m].C_x + detect[m].C_width / 2)
					&& (detect[m].C_x + detect[m].C_width / 2) <= (detect[n].C_x + detect[n].C_width / 2) + 100
					&&
					(detect[n].C_y + detect[n].C_height / 2) - 100 <= (detect[m].C_y + detect[m].C_height / 2)
					&& (detect[m].C_y + detect[m].C_height / 2) <= (detect[n].C_y + detect[n].C_height / 2) + 100) {

					detect[m].territory_num = detect[n].territory_num;
				}
			}
		}
		Detect_Place det_FD[50];
		int FD_count = 0;
		//統一領域ごとに検出結果の表示
		for (int i = 1; i <= t_num; i++) {
			int final_num = 0;
			float fyudo = 0, cyudo = 0;
			int area = 0;
			for (int k = 0; detect[k].C_yudo != 0; k++) {
				if (detect[k].territory_num == i && detect[k].F_yudo > fyudo) {
					final_num = k;
					fyudo = detect[k].F_yudo;
				}
				/*	if (detect[k].territory_num == i) {
				if ((detect[k].F_width * detect[k].F_height) > area) {
				final_num = k;
				fyudo = detect[k].F_yudo;
				area = detect[k].F_width * detect[k].F_height;
				}
				else if ((detect[k].F_width*detect[k].F_height) == area && detect[k].F_yudo > fyudo) {
				final_num = k;
				fyudo = detect[k].F_yudo;
				}
				}*/
			}
			det_FD[FD_count] = detect[final_num];
			det_FD[FD_count].territory_num = -1;
			FD_count++;
		}

		t_num = 0;
		for (int n = 0; det_FD[n].C_yudo != 0; n++) {
			if (det_FD[n].F_yudo == 0) continue;
			if (det_FD[n].territory_num == -1) {
				t_num++;
				det_FD[n].territory_num = t_num;
			}
			for (int m = n + 1; det_FD[m].C_yudo != 0; m++) {
				if (det_FD[m].F_yudo == 0) continue;

				if ((det_FD[n].C_x + det_FD[n].C_width / 2) - 100 <= (det_FD[m].C_x + det_FD[m].C_width / 2)
					&& (det_FD[m].C_x + det_FD[m].C_width / 2) <= (det_FD[n].C_x + det_FD[n].C_width / 2) + 100
					&&
					(det_FD[n].C_y + det_FD[n].C_height / 2) - 100 <= (det_FD[m].C_y + det_FD[m].C_height / 2)
					&& (det_FD[m].C_y + det_FD[m].C_height / 2) <= (det_FD[n].C_y + det_FD[n].C_height / 2) + 100) {

					det_FD[m].territory_num = det_FD[n].territory_num;
				}
			}
		}

		//統一領域ごとに検出結果の表示
		for (int i = 1; i <= t_num; i++) {
			int final_num = 0;
			float fyudo = 0, cyudo = 0;
			int area = 0;
			for (int k = 0; det_FD[k].C_yudo != 0; k++) {
				if (det_FD[k].territory_num == i && det_FD[k].F_yudo > fyudo) {
					final_num = k;
					fyudo = det_FD[k].F_yudo;
				}
				/*
				if (det_FD[k].territory_num == i) {
				if ((det_FD[k].F_width * det_FD[k].F_height) > area) {
				final_num = k;
				fyudo = det_FD[k].F_yudo;
				area = det_FD[k].F_width * det_FD[k].F_height;
				}
				else if ((det_FD[k].F_width*det_FD[k].F_height) == area && det_FD[k].F_yudo > fyudo) {
				final_num = k;
				fyudo = det_FD[k].F_yudo;
				}
				}
				*/

			}

			//矩形表示
			//FD結果をテキストファイルに保存
			fprintf_s(result_data, "%f", det_FD[final_num].C_yudo);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", det_FD[final_num].C_x);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", det_FD[final_num].C_y);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", det_FD[final_num].C_width);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", det_FD[final_num].C_height);
			fprintf_s(result_data, "\n");

			fprintf_s(result_data, "%f", det_FD[final_num].F_yudo);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", det_FD[final_num].F_x);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", det_FD[final_num].F_y);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", det_FD[final_num].F_width);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", det_FD[final_num].F_height);
			fprintf_s(result_data, "\n");

			printf("%f, %d, %d, %d, %d\n", det_FD[final_num].F_yudo, det_FD[final_num].F_x, det_FD[final_num].F_y, det_FD[final_num].F_width, det_FD[final_num].F_height);

			//	ans_img_CF = draw_rectangle(ans_img_CF, det_FD[final_num].C_x, det_FD[final_num].C_y, det_FD[final_num].C_width, det_FD[final_num].C_height, 255, 0, 0);
			//	ans_img_CF = draw_rectangle(ans_img_CF, det_FD[final_num].F_x, det_FD[final_num].F_y, det_FD[final_num].F_width, det_FD[final_num].F_height, 0, 255, 0);

		}
		//		imshow("", ans_img_CF);
		//		cvWaitKey(0);
		fclose(result_data);
	}
	fclose(test_data);

}

void detect_2() {
	//変数宣言
	//	int x, y;

	int file_num = 1;

	int count = 0;
	int hog_dim;

	//CoarseDetectorの取り込み
//	if ((CD = svm_load_model("C:/model_file/pre_model/CD_123.model")) == 0)exit(1);

	if ((FD_24x72 = svm_load_model("C:/model_file/pre_model/FD_24x72.model")) == 0)exit(1);
	if ((FD_24x80 = svm_load_model("C:/model_file/pre_model/FD_24x80.model")) == 0)exit(1);
	if ((FD_24x88 = svm_load_model("C:/model_file/pre_model/FD_24x88.model")) == 0)exit(1);

	if ((FD_32x88 = svm_load_model("C:/model_file/pre_model/FD_32x88.model")) == 0)exit(1);
	if ((FD_32x96 = svm_load_model("C:/model_file/pre_model/FD_32x96.model")) == 0)exit(1);
	if ((FD_32x104 = svm_load_model("C:/model_file/pre_model/FD_32x104.model")) == 0)exit(1);
	if ((FD_32x112 = svm_load_model("C:/model_file/pre_model/FD_32x112.model")) == 0)exit(1);
	if ((FD_32x120 = svm_load_model("C:/model_file/pre_model/FD_32x120.model")) == 0)exit(1);

	if ((FD_40x72 = svm_load_model("C:/model_file/pre_model/FD_40x72.model")) == 0)exit(1);
	if ((FD_40x80 = svm_load_model("C:/model_file/pre_model/FD_40x80.model")) == 0)exit(1);
	if ((FD_40x112 = svm_load_model("C:/model_file/pre_model/FD_40x112.model")) == 0)exit(1);
	if ((FD_40x120 = svm_load_model("C:/model_file/pre_model/FD_40x120.model")) == 0)exit(1);
	if ((FD_40x128 = svm_load_model("C:/model_file/pre_model/FD_40x128.model")) == 0)exit(1);

	if ((FD_48x80 = svm_load_model("C:/model_file/pre_model/FD_48x80.model")) == 0)exit(1);
	if ((FD_48x88 = svm_load_model("C:/model_file/pre_model/FD_48x88.model")) == 0)exit(1);
	if ((FD_48x96 = svm_load_model("C:/model_file/pre_model/FD_48x96.model")) == 0)exit(1);
	if ((FD_48x128 = svm_load_model("C:/model_file/pre_model/FD_48x128.model")) == 0)exit(1);

	if ((FD_56x96 = svm_load_model("C:/model_file/pre_model/FD_56x96.model")) == 0)exit(1);
	if ((FD_56x104 = svm_load_model("C:/model_file/pre_model/FD_56x104.model")) == 0)exit(1);
	if ((FD_56x112 = svm_load_model("C:/model_file/pre_model/FD_56x112.model")) == 0)exit(1);

	if ((FD_64x104 = svm_load_model("C:/model_file/pre_model/FD_64x104.model")) == 0)exit(1);
	if ((FD_64x112 = svm_load_model("C:/model_file/pre_model/FD_64x112.model")) == 0)exit(1);
	if ((FD_64x120 = svm_load_model("C:/model_file/pre_model/FD_64x120.model")) == 0)exit(1);
	if ((FD_64x128 = svm_load_model("C:/model_file/pre_model/FD_64x128.model")) == 0)exit(1);

	if ((FD_72x16 = svm_load_model("C:/model_file/pre_model/FD_72x16.model")) == 0)exit(1);
	if ((FD_72x120 = svm_load_model("C:/model_file/pre_model/FD_72x120.model")) == 0)exit(1);
	if ((FD_72x128 = svm_load_model("C:/model_file/pre_model/FD_72x128.model")) == 0)exit(1);

	if ((FD_80x16 = svm_load_model("C:/model_file/pre_model/FD_80x16.model")) == 0)exit(1);
	if ((FD_80x24 = svm_load_model("C:/model_file/pre_model/FD_80x24.model")) == 0)exit(1);
	if ((FD_80x128 = svm_load_model("C:/model_file/pre_model/FD_80x128.model")) == 0)exit(1);

	if ((FD_88x24 = svm_load_model("C:/model_file/pre_model/FD_88x24.model")) == 0)exit(1);
	if ((FD_88x32 = svm_load_model("C:/model_file/pre_model/FD_88x32.model")) == 0)exit(1);

	if ((FD_96x24 = svm_load_model("C:/model_file/pre_model/FD_96x24.model")) == 0)exit(1);
	if ((FD_96x32 = svm_load_model("C:/model_file/pre_model/FD_96x32.model")) == 0)exit(1);

	if ((FD_104x24 = svm_load_model("C:/model_file/pre_model/FD_104x24.model")) == 0)exit(1);
	if ((FD_104x32 = svm_load_model("C:/model_file/pre_model/FD_104x32.model")) == 0)exit(1);
	if ((FD_104x40 = svm_load_model("C:/model_file/pre_model/FD_104x40.model")) == 0)exit(1);

	if ((FD_112x24 = svm_load_model("C:/model_file/pre_model/FD_112x24.model")) == 0)exit(1);
	if ((FD_112x32 = svm_load_model("C:/model_file/pre_model/FD_112x32.model")) == 0)exit(1);
	if ((FD_112x40 = svm_load_model("C:/model_file/pre_model/FD_112x40.model")) == 0)exit(1);

	if ((FD_120x24 = svm_load_model("C:/model_file/pre_model/FD_120x24.model")) == 0)exit(1);
	if ((FD_120x32 = svm_load_model("C:/model_file/pre_model/FD_120x32.model")) == 0)exit(1);
	if ((FD_120x40 = svm_load_model("C:/model_file/pre_model/FD_120x40.model")) == 0)exit(1);

	if ((FD_128x32 = svm_load_model("C:/model_file/pre_model/FD_128x32.model")) == 0)exit(1);
	if ((FD_128x40 = svm_load_model("C:/model_file/pre_model/FD_128x40.model")) == 0)exit(1);
	if ((FD_128x48 = svm_load_model("C:/model_file/pre_model/FD_128x48.model")) == 0)exit(1);



	//テスト画像ファイル一覧メモ帳読み込み
	char test_name[1024], result_name[1024];
	FILE *test_data, *result_data;
	if (fopen_s(&test_data, "c:/photo/test_list_2.txt", "r") != 0) {
		cout << "missing" << endl;
		return;
	}

	while (fgets(test_name, 256, test_data) != NULL) {
		string name_tes = test_name;
		char new_test_name[1024];
		for (int i = 0; i < name_tes.length() - 1; i++) {
			new_test_name[i] = test_name[i];
			new_test_name[i + 1] = '\0';
		}
		count = 0;

		char test_path[1024] = "C:/photo/test_data_from_demo/test_data/";
		strcat_s(test_path, new_test_name);

		for (int i = 0; i < 1024; i++) {
			if (new_test_name[i] == 'b') {
				new_test_name[i] = 't';
				new_test_name[i + 1] = 'x';
				new_test_name[i + 2] = 't';
				new_test_name[i + 3] = '\0';
				break;
			}
			else new_test_name[i] = new_test_name[i];
		}
		char result_path[1024] = "result_data/";
		strcat_s(result_path, new_test_name);

		if (fopen_s(&result_data, result_path, "w") != 0) {
			cout << "missing 2" << endl;
			return;
		}
		//画像の取り込み
		cv::Mat ans_img_CF = cv::imread(test_path, 1);	//検出する画像
		cv::Mat check_img = ans_img_CF.clone();

		cv::Mat img;			//検出矩形処理を施す画像
		cvtColor(ans_img_CF, img, CV_RGB2GRAY);

		cout << file_num << ":" << new_test_name << endl;
		file_num++;

		//Detect_Placeオブジェクトの作成
		

		FILE *CD_data;
		char CD_path[1024] = "C:/photo/result_data_from_demo/2018_01_13_OOP/result_data/";
		strcat_s(CD_path, new_test_name);
		cout << CD_path << endl;
		if (fopen_s(&CD_data, CD_path, "r") != 0) {
			cout << "not found CD result" << endl;
			return;
		}

		char tmp[5][1024];
		//CD検出結果の取り込み
		while (fgets(tmp[0], 256, CD_data) != NULL) {
			fgets(tmp[1], 256, CD_data);
			fgets(tmp[2], 256, CD_data);
			fgets(tmp[3], 256, CD_data);
			fgets(tmp[4], 256, CD_data);

			Detect_Place detect;

			detect.C_yudo = atof(tmp[0]);
			detect.C_x = atoi(tmp[1]);
			detect.C_y = atoi(tmp[2]);
			detect.C_width = atoi(tmp[3]);
			detect.C_height = atoi(tmp[4]);

			//Fine Detectorによる検出
			FD_max_ans = 0.0;
			FD_max_XY = 0;

			cv::Mat FD_img = img(cv::Rect(detect.C_x, detect.C_y, detect.C_width, detect.C_height));

			cv::resize(FD_img, FD_img, cv::Size(), 128.0 / FD_img.cols, 128.0 / FD_img.rows);

			FD_predict(24, 72, FD_img, FD_24x72);
			FD_predict(24, 80, FD_img, FD_24x80);
			FD_predict(24, 88, FD_img, FD_24x88);

			FD_predict(32, 88, FD_img, FD_32x88);
			FD_predict(32, 96, FD_img, FD_32x96);
			FD_predict(32, 104, FD_img, FD_32x104);
			FD_predict(32, 112, FD_img, FD_32x112);
			FD_predict(32, 120, FD_img, FD_32x120);

			FD_predict(40, 72, FD_img, FD_40x72);
			FD_predict(40, 80, FD_img, FD_40x80);
			FD_predict(40, 112, FD_img, FD_40x112);
			FD_predict(40, 120, FD_img, FD_40x120);
			FD_predict(40, 128, FD_img, FD_40x128);

			FD_predict(48, 80, FD_img, FD_48x80);
			FD_predict(48, 88, FD_img, FD_48x88);
			FD_predict(48, 96, FD_img, FD_48x96);
			FD_predict(48, 128, FD_img, FD_48x128);

			FD_predict(56, 96, FD_img, FD_56x96);
			FD_predict(56, 104, FD_img, FD_56x104);
			FD_predict(56, 112, FD_img, FD_56x112);

			FD_predict(64, 104, FD_img, FD_64x104);
			FD_predict(64, 112, FD_img, FD_64x112);
			FD_predict(64, 120, FD_img, FD_64x120);
			FD_predict(64, 128, FD_img, FD_64x128);

			FD_predict(72, 16, FD_img, FD_72x16);
			FD_predict(72, 120, FD_img, FD_72x120);
			FD_predict(72, 128, FD_img, FD_72x128);

			FD_predict(80, 16, FD_img, FD_80x16);
			FD_predict(80, 24, FD_img, FD_80x24);
			FD_predict(80, 128, FD_img, FD_80x128);

			FD_predict(88, 24, FD_img, FD_88x24);
			FD_predict(88, 32, FD_img, FD_88x32);

			FD_predict(96, 24, FD_img, FD_96x24);
			FD_predict(96, 32, FD_img, FD_96x32);

			FD_predict(104, 24, FD_img, FD_104x24);
			FD_predict(104, 32, FD_img, FD_104x32);
			FD_predict(104, 40, FD_img, FD_104x40);

			FD_predict(112, 24, FD_img, FD_112x24);
			FD_predict(112, 32, FD_img, FD_112x32);
			FD_predict(112, 40, FD_img, FD_112x40);

			FD_predict(120, 24, FD_img, FD_120x24);
			FD_predict(120, 32, FD_img, FD_120x32);
			FD_predict(120, 40, FD_img, FD_120x40);

			FD_predict(128, 32, FD_img, FD_128x32);
			FD_predict(128, 40, FD_img, FD_128x40);
			FD_predict(128, 48, FD_img, FD_128x48);


			if (FD_max_ans != 0) {
				int F_x1 = (detect.C_x + (float)detect.C_width / 2) - (((float)detect.C_width / 128) * (FD_max_XY / 2000));
				int F_y1 = (detect.C_y + (float)detect.C_height / 2) - (((float)detect.C_height / 128) * (FD_max_XY % 1000 / 2));
				int F_width = ((float)detect.C_width / 128)*(FD_max_XY / 1000);
				int F_height = ((float)detect.C_height / 128)*(FD_max_XY % 1000);

				detect.F_yudo = FD_max_ans;
				detect.F_x = F_x1;
				detect.F_y = F_y1;
				detect.F_width = F_width;
				detect.F_height = F_height;



				fprintf_s(result_data, "%f", detect.C_yudo);
				fprintf_s(result_data, "\n");
				fprintf_s(result_data, "%d", detect.C_x);
				fprintf_s(result_data, "\n");
				fprintf_s(result_data, "%d", detect.C_y);
				fprintf_s(result_data, "\n");
				fprintf_s(result_data, "%d", detect.C_width);
				fprintf_s(result_data, "\n");
				fprintf_s(result_data, "%d", detect.C_height);
				fprintf_s(result_data, "\n");

				fprintf_s(result_data, "%f", detect.F_yudo);
				fprintf_s(result_data, "\n");
				fprintf_s(result_data, "%d", detect.F_x);
				fprintf_s(result_data, "\n");
				fprintf_s(result_data, "%d", detect.F_y);
				fprintf_s(result_data, "\n");
				fprintf_s(result_data, "%d", detect.F_width);
				fprintf_s(result_data, "\n");
				fprintf_s(result_data, "%d", detect.F_height);
				fprintf_s(result_data, "\n");
			}
			//		cout << FD_max_ans << endl;
			//		printf("%f, %d, %d, %d, %d\n", detect.F_yudo, detect.F_x, detect.F_y, detect.F_width, detect.F_height);

			//	ans_img_CF = draw_rectangle(ans_img_CF, detect.C_x, detect.C_y, detect.C_width, detect.C_height, 255, 0, 0);
			//	ans_img_CF = draw_rectangle(ans_img_CF, detect.F_x, detect.F_y, detect.F_width, detect.F_height, 0, 255, 0);
			//	cv::imshow("", ans_img_CF);
			//	cvWaitKey(0);

		}
		fclose(CD_data);
//		cv::imshow("", ans_img_CF);
//		cvWaitKey(0);
	}
	count++;
	
	fclose(test_data);
}

void ROC_data() {
	
	int file_num = 1;
	int hog_dim;

	//CoarseDetectorの取り込み
	{
		if ((CD = svm_load_model("C:/model_file/pre_model/CD_128x128.model")) == 0)exit(1);

		if ((FD_24x72 = svm_load_model("C:/model_file/pre_model/FD_24x72.model")) == 0)exit(1);
		if ((FD_24x80 = svm_load_model("C:/model_file/pre_model/FD_24x80.model")) == 0)exit(1);
		if ((FD_24x88 = svm_load_model("C:/model_file/pre_model/FD_24x88.model")) == 0)exit(1);

		if ((FD_32x88 = svm_load_model("C:/model_file/pre_model/FD_32x88.model")) == 0)exit(1);
		if ((FD_32x96 = svm_load_model("C:/model_file/pre_model/FD_32x96.model")) == 0)exit(1);
		if ((FD_32x104 = svm_load_model("C:/model_file/pre_model/FD_32x104.model")) == 0)exit(1);
		if ((FD_32x112 = svm_load_model("C:/model_file/pre_model/FD_32x112.model")) == 0)exit(1);
		if ((FD_32x120 = svm_load_model("C:/model_file/pre_model/FD_32x120.model")) == 0)exit(1);

		if ((FD_40x72 = svm_load_model("C:/model_file/pre_model/FD_40x72.model")) == 0)exit(1);
		if ((FD_40x80 = svm_load_model("C:/model_file/pre_model/FD_40x80.model")) == 0)exit(1);
		if ((FD_40x112 = svm_load_model("C:/model_file/pre_model/FD_40x112.model")) == 0)exit(1);
		if ((FD_40x120 = svm_load_model("C:/model_file/pre_model/FD_40x120.model")) == 0)exit(1);
		if ((FD_40x128 = svm_load_model("C:/model_file/pre_model/FD_40x128.model")) == 0)exit(1);

		if ((FD_48x80 = svm_load_model("C:/model_file/pre_model/FD_48x80.model")) == 0)exit(1);
		if ((FD_48x88 = svm_load_model("C:/model_file/pre_model/FD_48x88.model")) == 0)exit(1);
		if ((FD_48x96 = svm_load_model("C:/model_file/pre_model/FD_48x96.model")) == 0)exit(1);
		if ((FD_48x128 = svm_load_model("C:/model_file/pre_model/FD_48x128.model")) == 0)exit(1);

		if ((FD_56x96 = svm_load_model("C:/model_file/pre_model/FD_56x96.model")) == 0)exit(1);
		if ((FD_56x104 = svm_load_model("C:/model_file/pre_model/FD_56x104.model")) == 0)exit(1);
		if ((FD_56x112 = svm_load_model("C:/model_file/pre_model/FD_56x112.model")) == 0)exit(1);

		if ((FD_64x104 = svm_load_model("C:/model_file/pre_model/FD_64x104.model")) == 0)exit(1);
		if ((FD_64x112 = svm_load_model("C:/model_file/pre_model/FD_64x112.model")) == 0)exit(1);
		if ((FD_64x120 = svm_load_model("C:/model_file/pre_model/FD_64x120.model")) == 0)exit(1);
		if ((FD_64x128 = svm_load_model("C:/model_file/pre_model/FD_64x128.model")) == 0)exit(1);

		if ((FD_72x16 = svm_load_model("C:/model_file/pre_model/FD_72x16.model")) == 0)exit(1);
		if ((FD_72x120 = svm_load_model("C:/model_file/pre_model/FD_72x120.model")) == 0)exit(1);
		if ((FD_72x128 = svm_load_model("C:/model_file/pre_model/FD_72x128.model")) == 0)exit(1);

		if ((FD_80x16 = svm_load_model("C:/model_file/pre_model/FD_80x16.model")) == 0)exit(1);
		if ((FD_80x24 = svm_load_model("C:/model_file/pre_model/FD_80x24.model")) == 0)exit(1);
		if ((FD_80x128 = svm_load_model("C:/model_file/pre_model/FD_80x128.model")) == 0)exit(1);

		if ((FD_88x24 = svm_load_model("C:/model_file/pre_model/FD_88x24.model")) == 0)exit(1);
		if ((FD_88x32 = svm_load_model("C:/model_file/pre_model/FD_88x32.model")) == 0)exit(1);

		if ((FD_96x24 = svm_load_model("C:/model_file/pre_model/FD_96x24.model")) == 0)exit(1);
		if ((FD_96x32 = svm_load_model("C:/model_file/pre_model/FD_96x32.model")) == 0)exit(1);

		if ((FD_104x24 = svm_load_model("C:/model_file/pre_model/FD_104x24.model")) == 0)exit(1);
		if ((FD_104x32 = svm_load_model("C:/model_file/pre_model/FD_104x32.model")) == 0)exit(1);
		if ((FD_104x40 = svm_load_model("C:/model_file/pre_model/FD_104x40.model")) == 0)exit(1);

		if ((FD_112x24 = svm_load_model("C:/model_file/pre_model/FD_112x24.model")) == 0)exit(1);
		if ((FD_112x32 = svm_load_model("C:/model_file/pre_model/FD_112x32.model")) == 0)exit(1);
		if ((FD_112x40 = svm_load_model("C:/model_file/pre_model/FD_112x40.model")) == 0)exit(1);

		if ((FD_120x24 = svm_load_model("C:/model_file/pre_model/FD_120x24.model")) == 0)exit(1);
		if ((FD_120x32 = svm_load_model("C:/model_file/pre_model/FD_120x32.model")) == 0)exit(1);
		if ((FD_120x40 = svm_load_model("C:/model_file/pre_model/FD_120x40.model")) == 0)exit(1);

		if ((FD_128x32 = svm_load_model("C:/model_file/pre_model/FD_128x32.model")) == 0)exit(1);
		if ((FD_128x40 = svm_load_model("C:/model_file/pre_model/FD_128x40.model")) == 0)exit(1);
		if ((FD_128x48 = svm_load_model("C:/model_file/pre_model/FD_128x48.model")) == 0)exit(1);
	}

	//テスト画像ファイル一覧メモ帳読み込み
	char test_name[1024], result_name[1024];
	FILE *test_data, *result_data;
	if (fopen_s(&test_data, "predict-list.txt", "r") != 0) {
		cout << "missing" << endl;
		return;
	}

	while (fgets(test_name, 256, test_data) != NULL) {
		string name_tes = test_name;
		char new_test_name[1024];
		for (int i = 0; i < name_tes.length() - 1; i++) {
			new_test_name[i] = test_name[i];
			new_test_name[i + 1] = '\0';
		}

		char test_path[1024] = "C:/photo/test_data_from_demo/predict_data/";
		strcat_s(test_path, new_test_name);

		for (int i = 0; i < 1024; i++) {
			if (new_test_name[i] == 'b' && new_test_name[i+1]=='m') {
				new_test_name[i] = 't';
				new_test_name[i + 1] = 'x';
				new_test_name[i + 2] = 't';
				new_test_name[i + 3] = '\0';
				break;
			}
			else new_test_name[i] = new_test_name[i];
		}
		char result_path[1024] = "result_data/";
		strcat_s(result_path, new_test_name);

		if (fopen_s(&result_data, result_path, "w") != 0) {
			cout << "missing 2" << endl;
			return;
		}
		//画像の取り込み
		cv::Mat ans_img_CF = cv::imread(test_path, 1);	//検出する画像

		cv::Mat img;			//検出矩形処理を施す画像
		cvtColor(ans_img_CF, img, CV_RGB2GRAY);

		cout << file_num << ":" << new_test_name << endl;
		file_num++;

		//Detect_Placeオブジェクトの作成
		Detect_Place detect;

		hog_dim = dimension(img.cols, img.rows);
		float hog_vector[50000];						//各次元のHOGを格納
		get_HOG(img, hog_vector);	//HOGの取得
		double ans = predict(hog_vector, hog_dim, CD);	//尤度の算出

		detect.C_yudo = ans;

		//Fine Detectorによる検出
		{
			FD_max_ans = 0.0;
			FD_max_XY = 0;
			FD_predict(24, 72, img, FD_24x72);
			FD_predict(24, 80, img, FD_24x80);
			FD_predict(24, 88, img, FD_24x88);

			FD_predict(32, 88, img, FD_32x88);
			FD_predict(32, 96, img, FD_32x96);
			FD_predict(32, 104, img, FD_32x104);
			FD_predict(32, 112, img, FD_32x112);
			FD_predict(32, 120, img, FD_32x120);

			FD_predict(40, 72, img, FD_40x72);
			FD_predict(40, 80, img, FD_40x80);
			FD_predict(40, 112, img, FD_40x112);
			FD_predict(40, 120, img, FD_40x120);
			FD_predict(40, 128, img, FD_40x128);

			FD_predict(48, 80, img, FD_48x80);
			FD_predict(48, 88, img, FD_48x88);
			FD_predict(48, 96, img, FD_48x96);
			FD_predict(48, 128, img, FD_48x128);

			FD_predict(56, 96, img, FD_56x96);
			FD_predict(56, 104, img, FD_56x104);
			FD_predict(56, 112, img, FD_56x112);

			FD_predict(64, 104, img, FD_64x104);
			FD_predict(64, 112, img, FD_64x112);
			FD_predict(64, 120, img, FD_64x120);
			FD_predict(64, 128, img, FD_64x128);

			FD_predict(72, 16, img, FD_72x16);
			FD_predict(72, 120, img, FD_72x120);
			FD_predict(72, 128, img, FD_72x128);

			FD_predict(80, 16, img, FD_80x16);
			FD_predict(80, 24, img, FD_80x24);
			FD_predict(80, 128, img, FD_80x128);

			FD_predict(88, 24, img, FD_88x24);
			FD_predict(88, 32, img, FD_88x32);

			FD_predict(96, 24, img, FD_96x24);
			FD_predict(96, 32, img, FD_96x32);

			FD_predict(104, 24, img, FD_104x24);
			FD_predict(104, 32, img, FD_104x32);
			FD_predict(104, 40, img, FD_104x40);

			FD_predict(112, 24, img, FD_112x24);
			FD_predict(112, 32, img, FD_112x32);
			FD_predict(112, 40, img, FD_112x40);

			FD_predict(120, 24, img, FD_120x24);
			FD_predict(120, 32, img, FD_120x32);
			FD_predict(120, 40, img, FD_120x40);

			FD_predict(128, 32, img, FD_128x32);
			FD_predict(128, 40, img, FD_128x40);
			FD_predict(128, 48, img, FD_128x48);
		}

		detect.F_yudo = FD_max_ans;
		detect.F_x = (FD_max_XY / 1000) / 2;
		detect.F_y = (FD_max_XY % 1000) / 2;
		detect.F_width = FD_max_XY /1000;
		detect.F_height = FD_max_XY % 1000;

		//FD結果をテキストファイルに保存
		fprintf_s(result_data, "%f", detect.C_yudo);
		fprintf_s(result_data, "\n");

		fprintf_s(result_data, "%f", detect.F_yudo);
		fprintf_s(result_data, "\n");
		fprintf_s(result_data, "%d", detect.F_x);
		fprintf_s(result_data, "\n");
		fprintf_s(result_data, "%d", detect.F_y);
		fprintf_s(result_data, "\n");
		fprintf_s(result_data, "%d", detect.F_width);
		fprintf_s(result_data, "\n");
		fprintf_s(result_data, "%d", detect.F_height);
		fprintf_s(result_data, "\n");

		printf("%s, %f, %f\n", new_test_name, detect.C_yudo, detect.F_yudo);

		fclose(result_data);
	}
		fclose(test_data);
}

int main(int argc, char** argv) {
	
	ROC_data();

	return 0;
}
